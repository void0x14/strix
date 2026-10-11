"""Model-outage failover: primary slug → emergency slug → checkpoint + notify.

Every model call in a scan goes through one provider, so a provider that cannot
reach the primary model must be able to move *every* call to the emergency model
without the scan restarting. This module holds that state machine and the thin
model wrapper that applies it.

The ladder is fixed:

* 10 consecutive failures to reach the primary model switch every call to the
  emergency slug, on the *same* endpoint (the emergency model is only a
  different slug — the base URL never changes, so it can never drift to a direct
  provider host).
* 10 consecutive failures on the emergency model too mean the whole route is
  down: the scan checkpoints everything already on disk and notifies the
  operator.

Each attempt is bounded by :data:`ATTEMPT_WINDOW_S` (~30 s), so ten attempts
span roughly five minutes. The clock is injected, so a test walks the whole
ladder without waiting real seconds.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from agents.models.interface import Model


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from agents.items import ModelResponse, TResponseStreamEvent


logger = logging.getLogger(__name__)


# Attempts on one model before moving down the ladder.
PRIMARY_FAILURES_BEFORE_SWITCH = 10
EMERGENCY_FAILURES_BEFORE_STOP = 10

# Upper bound on a single attempt: the retry happens inside this window, so ten
# attempts take about five minutes.
ATTEMPT_WINDOW_S = 30.0

Route = Literal["primary", "emergency"]


@dataclass(frozen=True, slots=True)
class FailoverDecision:
    """What the ladder decided after one model failure."""

    route: Route
    switched: bool
    exhausted: bool
    failures: int
    elapsed_s: float


@dataclass(slots=True)
class FailoverState:
    """Mutable ladder state, snapshotted so a resumed scan keeps its place."""

    route: Route = "primary"
    failures: int = 0
    first_failure_at: float | None = None
    switches: int = 0


class FailoverController:
    """Track consecutive failures and decide the ladder step.

    Pure: the caller passes ``now`` and performs the I/O. ``on_exhausted`` is
    invoked once, the moment the emergency route also runs out.
    """

    def __init__(
        self,
        *,
        primary: str,
        emergency: str,
        primary_failures_before_switch: int = PRIMARY_FAILURES_BEFORE_SWITCH,
        emergency_failures_before_stop: int = EMERGENCY_FAILURES_BEFORE_STOP,
        attempt_window_s: float = ATTEMPT_WINDOW_S,
        state: FailoverState | None = None,
    ) -> None:
        self.primary = primary
        self.emergency = emergency
        self._switch_after = primary_failures_before_switch
        self._stop_after = emergency_failures_before_stop
        self.attempt_window_s = attempt_window_s
        self.state = state or FailoverState()

    @property
    def route(self) -> Route:
        return self.state.route

    def active_model(self) -> str:
        return self.primary if self.state.route == "primary" else self.emergency

    def next_attempt_delay(self, attempt: int) -> float:
        """Backoff before attempt ``attempt`` (1-based), never past the window."""
        if attempt <= 1:
            return 0.0
        return min(float(2 ** (attempt - 1)), self.attempt_window_s)

    def record_failure(self, now: float) -> FailoverDecision:
        """Record one failed attempt and return the resulting decision."""
        if self.state.failures == 0:
            self.state.first_failure_at = now
        self.state.failures += 1
        elapsed = now - (self.state.first_failure_at or now)
        threshold = self._switch_after if self.state.route == "primary" else self._stop_after
        if self.state.failures < threshold:
            return FailoverDecision(
                route=self.state.route,
                switched=False,
                exhausted=False,
                failures=self.state.failures,
                elapsed_s=elapsed,
            )
        if self.state.route == "primary":
            self.state.route = "emergency"
            self.state.failures = 0
            self.state.first_failure_at = None
            self.state.switches += 1
            logger.warning(
                "primary model %s failed %d consecutive attempts; switching every call "
                "to emergency model %s on the same endpoint",
                self.primary,
                threshold,
                self.emergency,
            )
            return FailoverDecision(
                route="emergency",
                switched=True,
                exhausted=False,
                failures=threshold,
                elapsed_s=elapsed,
            )
        logger.error(
            "emergency model %s failed %d consecutive attempts; the model route is down",
            self.emergency,
            threshold,
        )
        return FailoverDecision(
            route="emergency",
            switched=False,
            exhausted=True,
            failures=threshold,
            elapsed_s=elapsed,
        )

    def record_success(self) -> None:
        """A completed call is real progress: the failure streak resets."""
        if self.state.failures or self.state.first_failure_at is not None:
            logger.info(
                "model route %s recovered after %d failure(s)",
                self.state.route,
                self.state.failures,
            )
        self.state.failures = 0
        self.state.first_failure_at = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "route": self.state.route,
            "failures": self.state.failures,
            "switches": self.state.switches,
        }

    def restore(self, snap: dict[str, Any]) -> None:
        route = snap.get("route")
        self.state = FailoverState(
            route="emergency" if route == "emergency" else "primary",
            failures=_as_int(snap.get("failures")),
            switches=_as_int(snap.get("switches")),
        )


def _as_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


class ModelOutageError(RuntimeError):
    """Both the primary and the emergency model route are unreachable."""


class FailoverModel(Model):
    """Delegate to the controller's active model, counting failures.

    A failure is any exception out of the inner model call. On the failure that
    crosses the emergency threshold the wrapper raises :class:`ModelOutageError`
    so the runner can checkpoint and notify.
    """

    def __init__(
        self,
        *,
        controller: FailoverController,
        resolve: Callable[[str], Model],
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], Any] | None = None,
        on_exhausted: Callable[[FailoverDecision], Any] | None = None,
    ) -> None:
        self._controller = controller
        self._resolve = resolve
        self._clock = clock or _monotonic
        self._sleep = sleep or asyncio.sleep
        self._on_exhausted = on_exhausted
        self._models: dict[str, Model] = {}
        self._lock = asyncio.Lock()

    def _model_for(self, slug: str) -> Model:
        model = self._models.get(slug)
        if model is None:
            model = self._resolve(slug)
            self._models[slug] = model
        return model

    def seed(self, slug: str, model: Model) -> None:
        """Reuse an already-resolved model for ``slug`` instead of rebuilding it."""
        self._models.setdefault(slug, model)

    async def _call(self, operation: str, invoke: Callable[[Model], Awaitable[Any]]) -> Any:
        attempt = 0
        while True:
            slug = self._controller.active_model()
            model = self._model_for(slug)
            try:
                result = await invoke(model)
            except Exception as exc:
                if isinstance(exc, asyncio.CancelledError):
                    raise
                decision = self._controller.record_failure(self._clock())
                logger.warning(
                    "model %s %s failed (attempt %d on %s, elapsed %.0fs): %r",
                    slug,
                    operation,
                    decision.failures,
                    decision.route,
                    decision.elapsed_s,
                    exc,
                )
                if decision.exhausted:
                    await self._exhausted(decision)
                    raise ModelOutageError(
                        f"both the primary model {self._controller.primary} and the "
                        f"emergency model {self._controller.emergency} are unreachable"
                    ) from exc
                attempt += 1
                await self._sleep(self._controller.next_attempt_delay(attempt))
                continue
            self._controller.record_success()
            return result

    async def _exhausted(self, decision: FailoverDecision) -> None:
        if self._on_exhausted is None:
            return
        try:
            result = self._on_exhausted(decision)
            if hasattr(result, "__await__"):
                await result
        except Exception:
            logger.exception("model-outage notification callback failed")

    async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:
        async def _invoke(model: Model) -> Any:
            return await model.get_response(*args, **kwargs)

        result = await self._call("get_response", _invoke)
        return cast("ModelResponse", result)

    async def stream_response(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[TResponseStreamEvent]:
        # A streaming call must be consumed inside the retry boundary, so the
        # events are buffered per attempt and replayed; the SDK's stream is
        # consumed eagerly by the run loop, so buffering costs only memory
        # proportional to one turn's events.
        async def _invoke(model: Model) -> Any:
            return await _collect(model.stream_response(*args, **kwargs))

        events = await self._call("stream_response", _invoke)
        for event in cast("list[TResponseStreamEvent]", events):
            yield event

    async def close(self) -> None:
        for model in self._models.values():
            close = getattr(model, "close", None)
            if close is not None:
                try:
                    await close()
                except Exception:  # noqa: BLE001 - closing a model is best-effort
                    logger.debug("closing inner model failed", exc_info=True)

    def get_retry_advice(self, request: Any) -> Any:
        return self._model_for(self._controller.active_model()).get_retry_advice(request)


async def _collect(stream: AsyncIterator[TResponseStreamEvent]) -> list[TResponseStreamEvent]:
    return [event async for event in stream]


class FailoverModelProvider:
    """Wrap a delegate provider so every resolved model carries failover.

    ``RunConfig.model_provider`` is one object shared by the root agent and every
    sub-agent, so wrapping it here moves *every* model call down the ladder at
    once — there is no per-agent switch to miss. One wrapper is handed out for
    every requested name; it picks the active slug per attempt, so the emergency
    model is resolved through the delegate and inherits the primary's endpoint
    configuration (credentials and base URL) exactly.
    """

    def __init__(
        self,
        delegate: Any,
        controller: FailoverController,
        *,
        on_exhausted: Callable[[FailoverDecision], Any] | None = None,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], Any] | None = None,
    ) -> None:
        self._delegate = delegate
        self._controller = controller
        self._wrapper = FailoverModel(
            controller=controller,
            resolve=delegate.get_model,
            clock=clock,
            sleep=sleep,
            on_exhausted=on_exhausted,
        )

    def get_model(self, model_name: str | None) -> Model:
        if model_name is not None:
            # Reuse the delegate's own resolution for the requested name so any
            # name-specific route (a subscription slug, a secondary endpoint) is
            # preserved; the controller's active slug drives what is called.
            self._wrapper.seed(model_name, self._delegate.get_model(model_name))
        return self._wrapper


def _monotonic() -> float:
    return time.monotonic()


# --- operator notification ---------------------------------------------------

_NOTIFICATION_TIMEOUT_S = 15


def notify_operator(
    title: str,
    message: str,
    *,
    runner: Callable[..., Any] | None = None,
) -> bool:
    """Send a desktop notification; return whether one was delivered.

    Uses the platform's own notifier and never raises: a notification failing
    must not be the thing that loses the run's state. ``runner`` is injectable so
    a test asserts the exact command without touching the desktop.
    """
    command = _notification_command(title, message)
    if command is None:
        logger.warning("no desktop notifier available; notification: %s - %s", title, message)
        return False
    execute = runner or subprocess.run
    try:
        result = execute(
            command,
            check=False,
            capture_output=True,
            timeout=_NOTIFICATION_TIMEOUT_S,
        )
    except Exception:
        logger.exception("desktop notification failed")
        return False
    delivered = getattr(result, "returncode", 0) == 0
    if not delivered:
        logger.warning("desktop notification returned %s", getattr(result, "returncode", "?"))
    return delivered


def _notification_command(title: str, message: str) -> list[str] | None:
    if sys.platform == "darwin":
        script = f"display notification {_osascript(message)} with title {_osascript(title)}"
        return ["osascript", "-e", script]
    if sys.platform.startswith("linux"):
        has_session = bool(os.environ.get("DISPLAY") or os.environ.get("DBUS_SESSION_BUS_ADDRESS"))
        if has_session and shutil.which("notify-send"):
            return ["notify-send", "-u", "critical", title, message]
        if shutil.which("zenity"):
            return ["zenity", "--notification", f"--text={title}: {message}"]
    return None


def _osascript(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


__all__ = [
    "ATTEMPT_WINDOW_S",
    "EMERGENCY_FAILURES_BEFORE_STOP",
    "PRIMARY_FAILURES_BEFORE_SWITCH",
    "FailoverController",
    "FailoverDecision",
    "FailoverModel",
    "FailoverModelProvider",
    "FailoverState",
    "ModelOutageError",
    "notify_operator",
]
