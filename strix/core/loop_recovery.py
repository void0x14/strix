"""Thinking-loop detection and the interrupt → re-prompt → compact → new-session ladder.

Some models degenerate mid-scan into a *thinking loop*: the stream keeps emitting
tiny, near-identical fragments (``Let me wait``, ``Go``, ``Now``, ...) at high
rate while the agent makes no progress. The same burst pattern also appears
during ordinary work — a model that reasons in small chunks emits hundreds of
one-to-four character deltas in a couple of seconds — so fragment size alone
cannot separate the two.

What separates them is *duration*: ordinary bursts are short (well under the
detection window), a loop keeps the burst open indefinitely. The detector
therefore holds a burst open only while fragments keep arriving without an idle
gap, and reports a loop once the burst has outlived the window **and** carries
the loop signature: a large number of tiny fragments of which one exact text
repeats many times. A long, varied reasoning turn fails the repeat test and a
silent turn produces no fragments at all, so neither is mistaken for a loop.

Everything here is a pure, injectable unit: the clock and the sleep are passed
in, so the whole ladder can be asserted without spending real seconds.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from strix.core.sessions import open_agent_session


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from agents.memory import Session


logger = logging.getLogger(__name__)


# --- detection ---------------------------------------------------------------

# How long a fragment burst must stay open before it counts as a loop. Ordinary
# reasoning bursts finish well inside this; a loop never closes.
LOOP_WINDOW_S = 20.0

# A burst closes when no fragment arrives for this long. Beyond it the next
# fragment starts a fresh burst, so two unrelated bursts are never summed.
BURST_IDLE_GAP_S = 5.0

# A "tiny" fragment is a handful of characters — the shape a looping model emits.
TINY_FRAGMENT_MAX_CHARS = 24

# A burst must carry at least this many fragments before it is even considered.
LOOP_MIN_FRAGMENTS = 60

# ... of which at least this fraction must be tiny.
LOOP_MIN_TINY_FRACTION = 0.6

# ... and one exact fragment text must repeat at least this many times. This is
# the loop signature proper: varied reasoning never repeats one fragment verbatim
# hundreds of times.
LOOP_MIN_REPEATS = 12

# The vocabulary test: a looping model cycles through a handful of phrases, so
# the burst uses very few distinct characters. A model reasoning normally for the
# same span keeps introducing new words and blows past this. Either this test or
# an overwhelming repeat count is required, so a long, varied turn that happens to
# reuse one short word is not mistaken for a loop.
LOOP_MAX_DISTINCT_CHARS = 200
LOOP_DOMINANT_REPEATS = 100

# Interrupt + re-prompt attempts on one session before forcing a compaction.
INTERRUPTS_BEFORE_COMPACT = 3

LoopAction = Literal["interrupt", "compact", "replace"]


@dataclass(frozen=True, slots=True)
class LoopSignal:
    """A detected thinking loop, with the evidence that triggered it."""

    fragments: int
    tiny_fragments: int
    repeats: int
    distinct_chars: int
    duration_s: float

    def describe(self) -> str:
        return (
            f"{self.fragments} fragments ({self.tiny_fragments} tiny, "
            f"{self.distinct_chars} distinct chars), one repeated {self.repeats}x, "
            f"burst open {self.duration_s:.1f}s"
        )


class ThinkingLoopDetector:
    """Track a fragment burst and report when it has become a thinking loop.

    ``observe`` is called for every streamed text fragment; it returns a
    :class:`LoopSignal` on the fragment that crosses the threshold, and ``None``
    the rest of the time. ``note_progress`` closes the current burst: a tool
    call, a lifecycle event, or a completed turn is real work, so the burst
    counter must not carry across it.
    """

    def __init__(
        self,
        *,
        window_s: float = LOOP_WINDOW_S,
        idle_gap_s: float = BURST_IDLE_GAP_S,
        tiny_max_chars: int = TINY_FRAGMENT_MAX_CHARS,
        min_fragments: int = LOOP_MIN_FRAGMENTS,
        min_tiny_fraction: float = LOOP_MIN_TINY_FRACTION,
        min_repeats: int = LOOP_MIN_REPEATS,
        max_distinct_chars: int = LOOP_MAX_DISTINCT_CHARS,
        dominant_repeats: int = LOOP_DOMINANT_REPEATS,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._window_s = window_s
        self._idle_gap_s = idle_gap_s
        self._tiny_max_chars = tiny_max_chars
        self._min_fragments = min_fragments
        self._min_tiny_fraction = min_tiny_fraction
        self._min_repeats = min_repeats
        self._max_distinct_chars = max_distinct_chars
        self._dominant_repeats = dominant_repeats
        self._clock = clock or time.monotonic
        self._burst_start: float | None = None
        self._last_fragment_at: float | None = None
        self._fragments = 0
        self._tiny = 0
        self._texts: Counter[str] = Counter()
        self._chars: set[str] = set()

    @property
    def clock(self) -> Callable[[], float]:
        """The clock the detector reads; injected so tests need no real seconds."""
        return self._clock

    def reset(self) -> None:
        """Forget the current burst."""
        self._burst_start = None
        self._last_fragment_at = None
        self._fragments = 0
        self._tiny = 0
        self._texts.clear()
        self._chars.clear()

    def note_progress(self, now: float | None = None) -> None:
        """Close the burst: something other than fragment spam happened."""
        del now
        self.reset()

    def observe(self, now: float, text: str) -> LoopSignal | None:
        """Feed one streamed text fragment; return a signal when it is a loop."""
        if not text:
            return None
        if (
            self._burst_start is None
            or self._last_fragment_at is None
            or now - self._last_fragment_at > self._idle_gap_s
        ):
            self.reset()
            self._burst_start = now
        self._last_fragment_at = now
        self._fragments += 1
        self._chars.update(text)
        if len(text) <= self._tiny_max_chars:
            self._tiny += 1
            self._texts[text] += 1

        if not self._is_loop(now):
            return None
        repeats = self._texts.most_common(1)[0][1] if self._texts else 0
        return LoopSignal(
            fragments=self._fragments,
            tiny_fragments=self._tiny,
            repeats=repeats,
            distinct_chars=len(self._chars),
            duration_s=now - self._burst_start,
        )

    def _is_loop(self, now: float) -> bool:
        """Whether the open burst now carries every loop signature."""
        if self._burst_start is None or now - self._burst_start <= self._window_s:
            return False
        if self._fragments < self._min_fragments:
            return False
        if self._tiny / self._fragments < self._min_tiny_fraction:
            return False
        repeats = self._texts.most_common(1)[0][1] if self._texts else 0
        if repeats < self._min_repeats:
            return False
        return len(self._chars) <= self._max_distinct_chars or repeats >= self._dominant_repeats


# --- recovery ladder ---------------------------------------------------------

LOOP_CONTINUATION_PROMPT = (
    "Your previous turn stalled in a repetitive reasoning loop: the same short "
    "fragments were emitted over and over without any tool call and without "
    "reaching a conclusion. That turn has been interrupted. Stop repeating "
    "yourself, do not restate the same plan, and take one concrete action now: "
    "call exactly one tool that advances the current task. If you are blocked "
    "waiting on something, say what it is in one line and then call the "
    "appropriate tool (wait_for_agents, respond_to_user, or the execution tool). "
    "This is loop-recovery attempt {attempt}/{limit}."
)


@dataclass(slots=True)
class LoopRecoveryState:
    """Per-session tally of loop-recovery attempts."""

    interrupts: int = 0
    compactions: int = 0
    replacements: int = 0

    def snapshot(self) -> dict[str, int]:
        return {
            "interrupts": self.interrupts,
            "compactions": self.compactions,
            "replacements": self.replacements,
        }


@dataclass(frozen=True, slots=True)
class RecoveryOutcome:
    """What the ladder decided to do about one detected loop."""

    action: LoopAction
    attempt: int
    prompt: str = ""
    signal: LoopSignal | None = None


class LoopRecoveryController:
    """Decide the ladder step for a detected loop, given the session's history.

    ``on_loop`` is the whole decision, kept free of I/O so a test can walk the
    ladder (interrupt x3 -> compact -> replace) without a live session. The
    caller performs the corresponding action through :func:`apply_recovery`.
    """

    def __init__(
        self,
        *,
        interrupts_before_compact: int = INTERRUPTS_BEFORE_COMPACT,
        state: LoopRecoveryState | None = None,
    ) -> None:
        self._interrupts_before_compact = interrupts_before_compact
        self.state = state or LoopRecoveryState()

    def on_loop(self, signal: LoopSignal) -> RecoveryOutcome:
        """Return the next ladder step for a freshly detected loop."""
        if self.state.interrupts < self._interrupts_before_compact:
            self.state.interrupts += 1
            return RecoveryOutcome(
                action="interrupt",
                attempt=self.state.interrupts,
                prompt=LOOP_CONTINUATION_PROMPT.format(
                    attempt=self.state.interrupts,
                    limit=self._interrupts_before_compact,
                ),
                signal=signal,
            )
        if self.state.compactions < 1:
            self.state.compactions += 1
            return RecoveryOutcome(action="compact", attempt=self.state.interrupts, signal=signal)
        self.state.replacements += 1
        return RecoveryOutcome(
            action="replace",
            attempt=self.state.interrupts,
            prompt=LOOP_CONTINUATION_PROMPT.format(
                attempt=self.state.interrupts,
                limit=self._interrupts_before_compact,
            ),
            signal=signal,
        )

    def note_progress(self) -> None:
        """The session recovered; the ladder starts over."""
        self.state = LoopRecoveryState()

    def snapshot(self) -> dict[str, int]:
        return self.state.snapshot()

    def restore(self, snap: dict[str, Any]) -> None:
        self.state = LoopRecoveryState(
            interrupts=_as_int(snap.get("interrupts")),
            compactions=_as_int(snap.get("compactions")),
            replacements=_as_int(snap.get("replacements")),
        )


def _as_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


# --- adapter -----------------------------------------------------------------


@dataclass(slots=True)
class RecoveryAdapter:
    """The thin I/O layer the ladder calls.

    Every field is a callable so the pure ladder can be tested with fakes and
    the live run supplies the real coordinator primitives (cancel the stalled
    stream, queue a continuation prompt, force a compaction, open a fresh
    session).
    """

    cancel_stream: Callable[[], None]
    send_prompt: Callable[[str], Awaitable[bool]]
    compact: Callable[[], Awaitable[bool]]
    replace_session: Callable[[], Awaitable[Session | None]]
    clock: Callable[[], float] = field(default=time.monotonic)


async def apply_recovery(
    outcome: RecoveryOutcome,
    adapter: RecoveryAdapter,
) -> Session | None:
    """Perform ``outcome`` through ``adapter``; return the replacement session.

    Returns the new session for a ``replace`` step and ``None`` otherwise. The
    session swap happens last, so a failure in an earlier step never leaves the
    agent without a session.
    """
    if outcome.action == "interrupt":
        adapter.cancel_stream()
        await adapter.send_prompt(outcome.prompt)
        return None
    if outcome.action == "compact":
        adapter.cancel_stream()
        await adapter.compact()
        await adapter.send_prompt(outcome.prompt)
        return None
    adapter.cancel_stream()
    new_session = await adapter.replace_session()
    await adapter.send_prompt(outcome.prompt)
    return new_session


def open_replacement_session(session: Session, *, suffix: str = "r") -> Session | None:
    """Open a sibling session on the same database, carrying the old history.

    The replacement lives in the same ``agents.db`` so the run record, the
    viewer, and ``--resume`` keep working unchanged; only the session id
    differs. Returns ``None`` when the session exposes no usable database path
    (a test double, or an in-memory session), in which case the caller keeps the
    existing session and the interrupt path still applies.
    """
    db_path = getattr(session, "db_path", None)
    session_id = getattr(session, "session_id", None)
    if db_path is None or session_id is None:
        return None
    if str(db_path) == ":memory:":
        return None
    return open_agent_session(f"{session_id}-{suffix}", _as_path(db_path))


def _as_path(value: Any) -> Path:
    return Path(str(value))


__all__ = [
    "BURST_IDLE_GAP_S",
    "INTERRUPTS_BEFORE_COMPACT",
    "LOOP_CONTINUATION_PROMPT",
    "LOOP_MIN_FRAGMENTS",
    "LOOP_MIN_REPEATS",
    "LOOP_MIN_TINY_FRACTION",
    "LOOP_WINDOW_S",
    "TINY_FRAGMENT_MAX_CHARS",
    "LoopRecoveryController",
    "LoopRecoveryState",
    "LoopSignal",
    "RecoveryAdapter",
    "RecoveryOutcome",
    "ThinkingLoopDetector",
    "apply_recovery",
    "open_replacement_session",
]
