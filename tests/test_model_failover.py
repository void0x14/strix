"""Tests for the model-outage failover ladder.

The controller takes an injected clock and the wrapper an injected sleep, so the
whole ladder — 10 primary failures, switch, 10 emergency failures, checkpoint,
notify — is asserted without spending the real ~5 minutes.
"""

from __future__ import annotations

from typing import Any

import pytest

from strix.core.model_failover import (
    ATTEMPT_WINDOW_S,
    EMERGENCY_FAILURES_BEFORE_STOP,
    PRIMARY_FAILURES_BEFORE_SWITCH,
    FailoverController,
    FailoverModel,
    FailoverModelProvider,
    ModelOutageError,
    notify_operator,
)


PRIMARY = "openai/workbuddy/global:deepseek-v4.1-flash"
EMERGENCY = "openai/minimax-cn/MiniMax-M3.1-Flash-Preview"


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class _Model:
    """A model stub that fails or succeeds per script."""

    def __init__(self, *, failures: int = 0) -> None:
        self.failures = failures
        self.calls = 0

    async def get_response(self, *_args: Any, **_kwargs: Any) -> Any:
        self.calls += 1
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError("provider unreachable")
        return {"ok": True}

    def stream_response(self, *_args: Any, **_kwargs: Any) -> Any:
        self.calls += 1
        if self.failures > 0:
            self.failures -= 1

            async def _boom() -> Any:
                raise RuntimeError("provider unreachable")
                yield  # pragma: no cover

            return _boom()

        async def _ok() -> Any:
            yield "event"

        return _ok()

    def get_retry_advice(self, _request: Any) -> Any:
        return None


# --- controller -------------------------------------------------------------


def test_ten_primary_failures_switch_to_the_emergency_slug() -> None:
    controller = FailoverController(primary=PRIMARY, emergency=EMERGENCY)
    clock = _Clock()
    decisions = [controller.record_failure(clock()) for _ in range(PRIMARY_FAILURES_BEFORE_SWITCH)]
    assert [d.switched for d in decisions] == [False] * 9 + [True]
    assert decisions[-1].route == "emergency"
    assert controller.active_model() == EMERGENCY
    assert decisions[-1].failures == PRIMARY_FAILURES_BEFORE_SWITCH


def test_attempts_span_the_operator_window() -> None:
    """Each attempt is retried inside ~30 s, so ten attempts take about five minutes."""
    controller = FailoverController(primary=PRIMARY, emergency=EMERGENCY)
    delays = [
        controller.next_attempt_delay(i) for i in range(1, PRIMARY_FAILURES_BEFORE_SWITCH + 1)
    ]
    assert all(delay <= ATTEMPT_WINDOW_S for delay in delays)
    assert delays[-1] == ATTEMPT_WINDOW_S
    assert sum(delays) <= PRIMARY_FAILURES_BEFORE_SWITCH * ATTEMPT_WINDOW_S


def test_ten_emergency_failures_exhaust_the_route() -> None:
    controller = FailoverController(primary=PRIMARY, emergency=EMERGENCY)
    clock = _Clock()
    for _ in range(PRIMARY_FAILURES_BEFORE_SWITCH):
        controller.record_failure(clock())
    decisions = [controller.record_failure(clock()) for _ in range(EMERGENCY_FAILURES_BEFORE_STOP)]
    assert [d.exhausted for d in decisions] == [False] * 9 + [True]
    assert controller.active_model() == EMERGENCY


def test_success_resets_the_failure_streak() -> None:
    controller = FailoverController(primary=PRIMARY, emergency=EMERGENCY)
    clock = _Clock()
    for _ in range(9):
        controller.record_failure(clock())
    controller.record_success()
    # The streak restarted: nine more failures must not switch yet.
    for _ in range(9):
        assert controller.record_failure(clock()).switched is False
    assert controller.route == "primary"


def test_controller_state_survives_a_snapshot_round_trip() -> None:
    controller = FailoverController(primary=PRIMARY, emergency=EMERGENCY)
    clock = _Clock()
    for _ in range(PRIMARY_FAILURES_BEFORE_SWITCH):
        controller.record_failure(clock())
    snap = controller.snapshot()

    restored = FailoverController(primary=PRIMARY, emergency=EMERGENCY)
    restored.restore(snap)
    assert restored.snapshot() == snap
    assert restored.active_model() == EMERGENCY


# --- model wrapper ----------------------------------------------------------


@pytest.mark.asyncio
async def test_wrapper_retries_then_moves_every_call_to_the_emergency_model() -> None:
    controller = FailoverController(primary=PRIMARY, emergency=EMERGENCY)
    clock = _Clock()
    models = {PRIMARY: _Model(failures=PRIMARY_FAILURES_BEFORE_SWITCH), EMERGENCY: _Model()}
    exhausted: list[Any] = []

    async def _on_exhausted(decision: Any) -> None:
        exhausted.append(decision)

    model = FailoverModel(
        controller=controller,
        resolve=lambda slug: models[slug],
        clock=clock,
        sleep=_no_sleep,
        on_exhausted=_on_exhausted,
    )
    result = await model.get_response()
    assert result == {"ok": True}
    assert models[PRIMARY].calls == PRIMARY_FAILURES_BEFORE_SWITCH
    assert models[EMERGENCY].calls == 1
    assert controller.active_model() == EMERGENCY
    assert exhausted == []


@pytest.mark.asyncio
async def test_wrapper_raises_and_notifies_when_both_routes_are_down() -> None:
    controller = FailoverController(primary=PRIMARY, emergency=EMERGENCY)
    clock = _Clock()
    models = {
        PRIMARY: _Model(failures=PRIMARY_FAILURES_BEFORE_SWITCH),
        EMERGENCY: _Model(failures=EMERGENCY_FAILURES_BEFORE_STOP),
    }
    exhausted: list[Any] = []

    async def _on_exhausted(decision: Any) -> None:
        exhausted.append(decision)

    model = FailoverModel(
        controller=controller,
        resolve=lambda slug: models[slug],
        clock=clock,
        sleep=_no_sleep,
        on_exhausted=_on_exhausted,
    )
    with pytest.raises(ModelOutageError):
        await model.get_response()
    assert len(exhausted) == 1
    assert exhausted[0].exhausted is True
    assert models[PRIMARY].calls == PRIMARY_FAILURES_BEFORE_SWITCH
    assert models[EMERGENCY].calls == EMERGENCY_FAILURES_BEFORE_STOP


@pytest.mark.asyncio
async def test_streaming_call_is_also_failed_over() -> None:
    controller = FailoverController(primary=PRIMARY, emergency=EMERGENCY)
    clock = _Clock()
    models = {PRIMARY: _Model(failures=PRIMARY_FAILURES_BEFORE_SWITCH), EMERGENCY: _Model()}
    model = FailoverModel(
        controller=controller, resolve=lambda slug: models[slug], clock=clock, sleep=_no_sleep
    )
    events = [event async for event in model.stream_response()]
    assert events == ["event"]
    assert controller.active_model() == EMERGENCY


@pytest.mark.asyncio
async def test_provider_resolves_the_emergency_slug_through_the_same_delegate() -> None:
    """The emergency model is only a different slug on the same endpoint."""
    controller = FailoverController(primary=PRIMARY, emergency=EMERGENCY)
    requested: list[str] = []

    class _Delegate:
        def get_model(self, name: str | None) -> Any:
            requested.append(str(name))
            return _Model()

    provider = FailoverModelProvider(_Delegate(), controller, sleep=_no_sleep)
    model = provider.get_model(PRIMARY)
    assert await model.get_response() == {"ok": True}
    controller.state.route = "emergency"
    assert await model.get_response() == {"ok": True}
    assert requested == [PRIMARY, EMERGENCY]


async def _no_sleep(_delay: float) -> None:
    return None


# --- notification -----------------------------------------------------------


def test_notify_operator_runs_the_platform_notifier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("strix.core.model_failover.sys.platform", "linux")
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(
        "strix.core.model_failover.shutil.which",
        lambda name: "/usr/bin/notify-send" if name == "notify-send" else None,
    )
    seen: list[list[str]] = []

    class _Result:
        returncode = 0

    def _run(command: list[str], **_kwargs: Any) -> Any:
        seen.append(command)
        return _Result()

    assert notify_operator("title", "body", runner=_run) is True
    assert seen == [["notify-send", "-u", "critical", "title", "body"]]


def test_notify_operator_reports_failure_without_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("strix.core.model_failover.sys.platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS", raising=False)
    monkeypatch.setattr("strix.core.model_failover.shutil.which", lambda _name: None)
    assert notify_operator("title", "body") is False
