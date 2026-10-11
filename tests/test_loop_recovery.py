"""Tests for the thinking-loop recovery ladder.

The detector and the ladder are pure units with an injected clock, so the whole
ladder is asserted here without waiting real seconds. The fixtures replay the
shape of a real stalled turn (a burst of tiny repeated fragments held open past
the window) and of a real healthy turn (a short burst, or a long varied one).
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from agents.memory import SQLiteSession

from strix.core import execution
from strix.core.agents import AgentCoordinator
from strix.core.execution import (
    ThinkingLoopError,
    _run_stream_events,
    _text_fragment,
)
from strix.core.loop_recovery import (
    INTERRUPTS_BEFORE_COMPACT,
    LOOP_WINDOW_S,
    LoopRecoveryController,
    LoopSignal,
    RecoveryAdapter,
    ThinkingLoopDetector,
    apply_recovery,
    open_replacement_session,
)


# --- stream-event shape -----------------------------------------------------


class _RawEvent:
    """A raw stream event carrying a text or reasoning delta."""

    def __init__(self, delta_type: str, text: str) -> None:
        self.type = "raw_response_event"
        self.data = _RawData(delta_type, text)


class _RawData:
    def __init__(self, delta_type: str, text: str) -> None:
        self.type = delta_type
        self.delta = text


class _NonFragmentEvent:
    def __init__(self, event_type: str = "run_item_stream_event") -> None:
        self.type = event_type


def _loop_fragment_types() -> list[str]:
    return ["response.output_text.delta", "response.reasoning_summary_text.delta"]


def _fragment_event(delta_type: str, text: str) -> Any:
    return _RawEvent(delta_type, text)


def test_text_fragment_reads_text_and_reasoning_deltas() -> None:
    for delta_type in _loop_fragment_types():
        assert _text_fragment(_fragment_event(delta_type, "Let me wait")) == "Let me wait"
    # A non-delta raw event, and a non-raw event, carry no fragment.
    assert _text_fragment(_RawEvent("response.created", "")) is None
    assert _text_fragment(_NonFragmentEvent()) is None
    assert _text_fragment(_fragment_event("response.output_text.delta", "")) is None


# --- detection --------------------------------------------------------------


def test_healthy_short_burst_is_not_a_loop() -> None:
    """Ordinary reasoning emits hundreds of tiny fragments in a couple of seconds.

    A real turn measured on the primary model produced 472 deltas, all tiny,
    inside 2.4 s. The burst closes before the window, so nothing fires.
    """
    detector = ThinkingLoopDetector()
    fired = []
    now = 1000.0
    for i in range(600):
        now += 0.004
        if signal := detector.observe(now, ["Let me", " ", "wait", " now"][i % 4]):
            fired.append(signal)
    # The whole burst finished well inside the window.
    assert now - 1000.0 < LOOP_WINDOW_S
    assert fired == []


def test_long_varied_turn_is_not_a_loop() -> None:
    """A long turn that keeps producing new words is work, not a loop."""
    detector = ThinkingLoopDetector()
    fired = []
    now = 1000.0
    words = [f"word{i}" for i in range(4000)]
    for word in words:
        now += 0.02
        if signal := detector.observe(now, word):
            fired.append(signal)
    assert now - 1000.0 > LOOP_WINDOW_S
    assert fired == []


def test_one_long_silent_turn_is_not_a_loop() -> None:
    """A silent turn produces no fragments at all, so it can never trip."""
    detector = ThinkingLoopDetector()
    # No fragments for a long time, then the turn ends. Nothing observed.
    assert detector.observe(1000.0, "") is None
    assert detector.observe(1000.0 + 60.0, "") is None


def test_loop_pattern_fires_after_the_window() -> None:
    """A burst of repeated tiny fragments held open past the window is a loop."""
    detector = ThinkingLoopDetector()
    fired: list[LoopSignal] = []
    now = 1000.0
    phrase = ["Let me wait", ".", " Go", ".", " Now", "."]
    for i in range(3000):
        now += 0.01  # 30 s of spam
        if signal := detector.observe(now, phrase[i % len(phrase)]):
            fired.append(signal)
    assert fired, "the detector must fire on a sustained repeated-fragment burst"
    signal = fired[0]
    assert signal.duration_s > LOOP_WINDOW_S
    assert signal.repeats >= 12
    assert signal.tiny_fragments == signal.fragments


def test_loop_burst_is_not_summed_across_an_idle_gap() -> None:
    """Two ordinary bursts separated by a gap must never add up to a loop."""
    detector = ThinkingLoopDetector()
    fired = []
    now = 1000.0
    for _ in range(2):
        for i in range(400):
            now += 0.004
            if signal := detector.observe(now, ["Go", ".", "Now", "."][i % 4]):
                fired.append(signal)
        now += 30.0  # a real pause: the next burst starts fresh
    assert fired == []


def test_tool_call_closes_the_burst() -> None:
    """Real work between fragments resets the burst counter."""
    detector = ThinkingLoopDetector()
    fired = []
    now = 1000.0
    for _ in range(10):
        for i in range(100):
            now += 0.01
            if signal := detector.observe(now, ["Go", "."][i % 2]):
                fired.append(signal)
        detector.note_progress(now)  # a tool call happened
    assert fired == []


# --- ladder -----------------------------------------------------------------


def _signal() -> LoopSignal:
    return LoopSignal(
        fragments=500, tiny_fragments=500, repeats=200, distinct_chars=12, duration_s=30.0
    )


def test_ladder_interrupts_three_times_then_compacts_then_replaces() -> None:
    """The exact operator-specified order, walked end to end."""
    controller = LoopRecoveryController()
    actions = [controller.on_loop(_signal()).action for _ in range(6)]
    assert actions == [
        "interrupt",
        "interrupt",
        "interrupt",
        "compact",
        "replace",
        "replace",
    ]
    assert controller.state.interrupts == INTERRUPTS_BEFORE_COMPACT
    assert controller.state.compactions == 1


def test_interrupt_outcome_carries_a_fresh_continuation_prompt() -> None:
    controller = LoopRecoveryController()
    first = controller.on_loop(_signal())
    assert first.action == "interrupt"
    assert first.prompt
    assert "1/3" in first.prompt
    second = controller.on_loop(_signal())
    assert "2/3" in second.prompt


def test_ladder_resets_after_progress() -> None:
    controller = LoopRecoveryController()
    controller.on_loop(_signal())
    controller.on_loop(_signal())
    controller.on_loop(_signal())
    controller.note_progress()
    assert controller.on_loop(_signal()).action == "interrupt"


def test_ladder_state_survives_a_snapshot_round_trip() -> None:
    controller = LoopRecoveryController()
    for _ in range(4):
        controller.on_loop(_signal())
    snap = controller.snapshot()

    restored = LoopRecoveryController()
    restored.restore(snap)
    assert restored.snapshot() == snap
    # Already at 3 interrupts and 1 compaction: the next step replaces.
    assert restored.on_loop(_signal()).action == "replace"


@pytest.mark.asyncio
async def test_apply_recovery_runs_each_step_in_order() -> None:
    calls: list[str] = []
    new_session = object()

    async def _send(prompt: str) -> bool:
        calls.append(f"prompt:{prompt[:13]}")
        return True

    async def _compact() -> bool:
        calls.append("compact")
        return True

    async def _replace() -> Any:
        calls.append("replace")
        return new_session

    adapter = RecoveryAdapter(
        cancel_stream=lambda: calls.append("cancel"),
        send_prompt=_send,
        compact=_compact,
        replace_session=_replace,
    )

    controller = LoopRecoveryController()
    for expected in ("interrupt", "interrupt", "interrupt", "compact"):
        outcome = controller.on_loop(_signal())
        assert outcome.action == expected
        assert await apply_recovery(outcome, adapter) is None
    calls.clear()
    outcome = controller.on_loop(_signal())
    assert outcome.action == "replace"
    assert await apply_recovery(outcome, adapter) is new_session
    assert calls == ["cancel", "replace", "prompt:Your previous"]


@pytest.mark.asyncio
async def test_apply_recovery_interrupt_cancels_the_stalled_stream() -> None:
    """The stalled turn must actually be interrupted, not just re-prompted."""
    cancelled: list[bool] = []
    sent: list[str] = []

    async def _send(prompt: str) -> bool:
        sent.append(prompt)
        return True

    async def _compact() -> bool:
        return False

    async def _replace() -> Any:
        return None

    adapter = RecoveryAdapter(
        cancel_stream=lambda: cancelled.append(True),
        send_prompt=_send,
        compact=_compact,
        replace_session=_replace,
    )
    outcome = LoopRecoveryController().on_loop(_signal())
    await apply_recovery(outcome, adapter)
    assert cancelled == [True]
    assert sent and sent[0] == outcome.prompt


def test_open_replacement_session_is_none_for_a_session_without_a_db() -> None:
    class _Bare:
        pass

    assert open_replacement_session(cast("Any", _Bare())) is None


def test_open_replacement_session_shares_the_database(tmp_path: Any) -> None:
    db = tmp_path / "agents.db"
    original = SQLiteSession("root", db)
    replacement = open_replacement_session(original)
    assert replacement is not None
    assert replacement.session_id == "root-r"
    assert str(replacement.db_path) == str(db)
    original.close()
    replacement.close()


# --- stream integration -----------------------------------------------------


class _ScriptedStream:
    """A run stream that emits a scripted event sequence, then completes."""

    def __init__(self, events: list[Any]) -> None:
        self._events = events
        self.cancelled = False

    async def stream_events(self) -> Any:
        for event in self._events:
            yield event

    def cancel(self, mode: str = "immediate") -> None:  # noqa: ARG002
        self.cancelled = True


class _Clock:
    def __init__(self, start: float = 1000.0, step: float = 0.01) -> None:
        self.now = start
        self.step = step

    def __call__(self) -> float:
        value = self.now
        self.now += self.step
        return value


@pytest.mark.asyncio
async def test_stream_events_raises_and_cancels_on_a_loop() -> None:
    clock = _Clock()
    detector = ThinkingLoopDetector(clock=clock)
    phrase = ["Let me wait", ".", " Go", ".", " Now", "."]
    events = [
        _fragment_event("response.reasoning_summary_text.delta", phrase[i % 6]) for i in range(4000)
    ]
    stream = _ScriptedStream(events)

    seen = 0
    with pytest.raises(ThinkingLoopError) as excinfo:
        async for _ in _run_stream_events(stream, detector):
            seen += 1
    assert stream.cancelled is True
    assert seen > 0
    assert excinfo.value.signal.duration_s > LOOP_WINDOW_S


@pytest.mark.asyncio
async def test_stream_events_passes_a_healthy_turn_through_untouched() -> None:
    clock = _Clock(step=0.001)
    detector = ThinkingLoopDetector(clock=clock)
    events: list[Any] = [_NonFragmentEvent()]
    events += [_fragment_event("response.output_text.delta", f"token{i}") for i in range(200)]
    events.append(_NonFragmentEvent("run_item_stream_event"))
    stream = _ScriptedStream(events)

    seen = 0
    async for _ in _run_stream_events(stream, detector):
        seen += 1
    assert seen == len(events)
    assert stream.cancelled is False


@pytest.mark.asyncio
async def test_stream_events_without_a_detector_is_a_passthrough() -> None:
    stream = _ScriptedStream([_NonFragmentEvent(), _fragment_event("x", "y")])
    seen = 0
    async for _ in _run_stream_events(stream, None):
        seen += 1
    assert seen == 2


# --- ladder wired into the real agent loop ----------------------------------


class _ScriptedLoopCycle:
    """A run cycle that stalls for the first ``stalls`` calls, then completes.

    It raises the real :class:`ThinkingLoopError` the shipped stream observer
    raises, so the ladder under test is the one the run loop actually runs.
    """

    def __init__(self, coordinator: Any, agent_id: str, stalls: int) -> None:
        self._coordinator = coordinator
        self._agent_id = agent_id
        self._stalls = stalls
        self.calls: list[Any] = []

    async def __call__(self, *_args: Any, **kwargs: Any) -> Any:
        self.calls.append(kwargs.get("input_data"))
        if len(self.calls) <= self._stalls:
            stream = _ScriptedStream([])
            raise ThinkingLoopError(_signal(), stream)
        await self._coordinator.set_status(self._agent_id, "completed")
        return _Result()


class _Result:
    final_output = "done"


async def _drive_loop(
    coordinator: Any,
    agent_id: str,
    monkeypatch: pytest.MonkeyPatch,
    cycle: Any,
    *,
    interactive: bool,
    session: Any = None,
) -> Any:
    monkeypatch.setattr(execution, "_run_cycle", cycle)
    monkeypatch.setattr(execution, "_run_cycle_parked", cycle)
    return await execution._run_until_lifecycle(
        object(),
        coordinator,
        agent_id,
        initial_input=[],
        run_config=None,  # type: ignore[arg-type]
        context={"agent_id": agent_id, "parent_id": None},
        max_turns=5,
        session=session,
        interactive=interactive,
        event_sink=None,
        hooks=None,
    )


@pytest.mark.asyncio
async def test_loop_ladder_interrupts_three_times_then_compacts(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(a) three interrupt+re-prompt attempts, (b) then a forced compaction."""
    coordinator = AgentCoordinator()
    await coordinator.register("root", "strix", parent_id=None)
    session = SQLiteSession("root", tmp_path / "agents.db")
    await coordinator.attach_runtime("root", session=session)
    cycle = _ScriptedLoopCycle(coordinator, "root", stalls=5)
    compactions: list[bool] = []

    async def _compact(_agent: Any, _session: Any, _run_config: Any, *, force: bool) -> bool:
        compactions.append(force)
        return True

    monkeypatch.setattr(execution, "_compact_session", _compact)

    await _drive_loop(coordinator, "root", monkeypatch, cycle, interactive=False, session=session)

    # Three interrupts, then the fourth detection forces a compaction.
    assert compactions == [True]
    assert len(cycle.calls) == 6
    assert coordinator.statuses["root"] == "completed"
    session.close()


@pytest.mark.asyncio
async def test_loop_ladder_replaces_the_session_after_compaction_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(c) when compaction does not restore progress, a fresh session continues."""
    coordinator = AgentCoordinator()
    await coordinator.register("root", "strix", parent_id=None)
    cycle = _ScriptedLoopCycle(coordinator, "root", stalls=5)

    async def _compact(_agent: Any, _session: Any, _run_config: Any, *, force: bool) -> bool:
        del force
        return False

    monkeypatch.setattr(execution, "_compact_session", _compact)

    await _drive_loop(coordinator, "root", monkeypatch, cycle, interactive=False)

    # 3 interrupts + 1 compaction + 1 replacement = 5 stalled cycles, then the
    # sixth cycle (a fresh session) completes.
    assert len(cycle.calls) == 6
    assert coordinator.statuses["root"] == "completed"


@pytest.mark.asyncio
async def test_loop_ladder_preserves_prior_session_items_across_replacement(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The replacement session carries every prior item: nothing is lost."""
    coordinator = AgentCoordinator()
    await coordinator.register("root", "strix", parent_id=None)
    session = SQLiteSession("root", tmp_path / "agents.db")
    await session.add_items(
        [{"role": "user", "content": "task"}, {"role": "assistant", "content": "finding: XSS"}]
    )
    await coordinator.attach_runtime("root", session=session)
    cycle = _ScriptedLoopCycle(coordinator, "root", stalls=5)

    async def _compact(_agent: Any, _session: Any, _run_config: Any, *, force: bool) -> bool:
        del force
        return False

    monkeypatch.setattr(execution, "_compact_session", _compact)

    await _drive_loop(coordinator, "root", monkeypatch, cycle, interactive=False, session=session)

    async with coordinator._lock:
        current = coordinator.runtimes["root"].session
    assert current is not None
    assert current is not session
    contents = [str(item.get("content", "")) for item in await current.get_items()]
    assert any("finding: XSS" in text for text in contents)
    assert any("loop-recovery attempt" in text for text in contents)
    if current is not session:
        current.close()
    session.close()


@pytest.mark.asyncio
async def test_normal_long_turn_does_not_trigger_any_loop_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run that never stalls must not be interrupted, compacted, or replaced."""
    coordinator = AgentCoordinator()
    await coordinator.register("root", "strix", parent_id=None)
    cycle = _ScriptedLoopCycle(coordinator, "root", stalls=0)
    compactions: list[bool] = []
    replacements: list[bool] = []

    async def _compact(_agent: Any, _session: Any, _run_config: Any, *, force: bool) -> bool:
        compactions.append(force)
        return True

    def _replace(_session: Any) -> Any:
        replacements.append(True)
        return None

    monkeypatch.setattr(execution, "_compact_session", _compact)
    monkeypatch.setattr(execution, "open_replacement_session", _replace)

    await _drive_loop(coordinator, "root", monkeypatch, cycle, interactive=False)

    assert compactions == []
    assert replacements == []
    assert len(cycle.calls) == 1
