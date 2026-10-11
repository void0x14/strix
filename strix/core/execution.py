"""Execution loop for addressable SDK-backed Strix agents."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import re
import uuid
from collections.abc import Callable
from dataclasses import replace
from functools import cache
from typing import TYPE_CHECKING, Any, cast

from agents import ItemHelpers, MessageOutputItem, RunConfig, Runner
from agents.exceptions import AgentsException, MaxTurnsExceeded, UserError
from agents.sandbox.errors import ExecTransportError
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
)

from strix.config import codex
from strix.core.hooks import (
    BudgetExceededError,
    BudgetPausedError,
    SubagentBudgetReservedError,
)
from strix.core.inputs import child_initial_input
from strix.core.loop_recovery import (
    LoopRecoveryController,
    LoopSignal,
    RecoveryAdapter,
    ThinkingLoopDetector,
    apply_recovery,
    open_replacement_session,
)
from strix.core.sessions import (
    enforce_image_budget,
    open_agent_session,
    replace_session_items,
    scrub_images_from_items,
    seed_initial_input,
    strip_all_images_from_session,
)
from strix.llm import request_log
from strix.llm.compaction import is_context_overflow, maybe_compact


if TYPE_CHECKING:
    from pathlib import Path

    from agents.items import TResponseInputItem
    from agents.lifecycle import RunHooks
    from agents.memory import Session, SQLiteSession
    from agents.result import RunResultBase
    from agents.run_config import CallModelData, ModelInputData

    from strix.core.agents import AgentCoordinator, Status


logger = logging.getLogger(__name__)

StreamEventSink = Callable[[str, Any], None]

_INPUT_REJECTION_CODES = frozenset({400, 404, 422})
# Replies meaning "this model takes no images", not image errors in general: a
# context overflow that counts "image/vision expansion" must not match.
_IMAGE_REJECTION = re.compile(
    r"no endpoints found that support image input"  # OpenRouter
    r"|image_url is only supported by certain models"  # OpenAI
    r"|is not a multimodal model|at most 0 image\(s\)"  # vLLM
    r"|does not support image input"  # LiteLLM's Fireworks check
    r"|doesn't support the image field"  # Bedrock Converse
    r"|unknown variant `image_url`"  # DeepSeek, e.g. via Vercel AI Gateway
    r"|'[^']*image[^']*' functionality not supported",  # Vercel AI Gateway (AI SDK), unverified
    re.IGNORECASE,
)
_MAX_COMPACTIONS_PER_CYCLE = 2

# Ceiling on session replacements across one agent loop. Each replacement opens a
# session whose own recovery budget starts over, so without this a model that
# loops on every session would spin forever. Past it the agent settles the same
# way recovery exhaustion does: parks when interactive, fails when autonomous.
_MAX_SESSION_REPLACEMENTS = 5

_LOOP_RECOVERY_KEY = "loop_recovery"


@cache
def _teardown_sandbox_errors() -> tuple[type[BaseException], ...]:
    """Sandbox-gone errors, tolerated during shutdown.

    The Docker SDK is imported here rather than at module scope: it is only
    reachable with the Docker runtime backend, and importing it eagerly puts it
    on every launch's critical path.
    """
    from docker import errors as docker_errors  # type: ignore[import-untyped, unused-ignore]

    return (ExecTransportError, docker_errors.NotFound)


class ProviderRefusalError(AgentsException):
    """Raised when a provider returns a structured refusal instead of an exception."""


class ThinkingLoopError(Exception):
    """Raised when a turn's stream degenerates into a thinking loop.

    Carries the detector's evidence and the stalled stream so the recovery
    ladder can interrupt it and decide the next step. It is deliberately not an
    ``AgentsException``: it is an execution-control signal, not a model or
    provider failure, and must never be folded into the generic crash handling.
    """

    def __init__(self, signal: LoopSignal, stream: Any) -> None:
        super().__init__(f"thinking loop detected: {signal.describe()}")
        self.signal = signal
        self.stream = stream


def _cancel_stream(stream: Any) -> None:
    """Cancel a stalled stream, tolerating a stream that is already settled."""
    cancel = getattr(stream, "cancel", None)
    if cancel is None:
        return
    with contextlib.suppress(Exception):
        cancel(mode="immediate")


def _text_fragment(event: Any) -> str | None:
    """The text a raw stream event adds, or None when it adds none.

    Both the visible answer and the reasoning summary are observed: a looping
    model spams whichever channel it is currently on, and the operator's own
    symptom (tiny ``Let me wait`` / ``Go`` fragments) appears on both.
    """
    if getattr(event, "type", None) != "raw_response_event":
        return None
    data = getattr(event, "data", None)
    data_type = getattr(data, "type", "")
    if not isinstance(data_type, str) or "delta" not in data_type:
        return None
    delta = getattr(data, "delta", None)
    return delta if isinstance(delta, str) and delta else None


def _run_stream_events(stream: Any, detector: ThinkingLoopDetector | None) -> Any:
    """Iterate a run's events, feeding the loop detector and raising on a loop.

    Kept as a plain async generator so both the streaming loop and its tests use
    one implementation: an event that carries a fragment is observed, any other
    event (a tool call, a message, a completed response) counts as progress and
    closes the burst.
    """

    async def _iterate() -> Any:
        async for event in stream.stream_events():
            if detector is None:
                yield event
                continue
            fragment = _text_fragment(event)
            if fragment is None:
                detector.note_progress()
                yield event
                continue
            signal = detector.observe(detector.clock(), fragment)
            if signal is not None:
                _cancel_stream(stream)
                raise ThinkingLoopError(signal, stream)
            yield event

    return _iterate()


def _structured_provider_refusal(result: Any) -> str | None:
    for item in getattr(result, "new_items", ()) or ():
        raw_item = getattr(item, "raw_item", None)
        for content in getattr(raw_item, "content", ()) or ():
            if getattr(content, "type", None) != "refusal":
                continue
            refusal = getattr(content, "refusal", None)
            if isinstance(refusal, str) and refusal.strip():
                return refusal.strip()
            return "The model provider refused this request."
    return None


def _run_config_model(run_config: RunConfig) -> str | None:
    return run_config.model if isinstance(run_config.model, str) else None


def _agent_instructions(agent: Any) -> str:
    instructions = getattr(agent, "instructions", None)
    return instructions if isinstance(instructions, str) else ""


def _agent_tools_text(agent: Any) -> str:
    parts: list[str] = []
    for tool in getattr(agent, "tools", []) or []:
        name = getattr(tool, "name", "")
        description = getattr(tool, "description", "") or ""
        schema = getattr(tool, "params_json_schema", "") or ""
        parts.append(f"{name} {description} {schema}")
    return "\n".join(parts)


async def _compact_session(
    agent: Any, session: Session, run_config: RunConfig, *, force: bool
) -> bool:
    model = _run_config_model(run_config)
    if session is None or model is None:
        return False
    return await maybe_compact(
        session,
        model=model,
        instructions=_agent_instructions(agent),
        tools_text=_agent_tools_text(agent),
        force=force,
    )


_TEXT_ONLY_IMAGE_TEXT = "[error: this model cannot view images; use `snapshot -i` instead]"


def _with_image_scrub(run_config: RunConfig, context: dict[str, Any]) -> RunConfig:
    if context.get("supports_images", True):
        return run_config
    # Chain any filter already set; it sees the scrubbed input.
    inner = run_config.call_model_input_filter

    async def _scrub(data: CallModelData[Any]) -> ModelInputData:
        model_data = replace(
            data.model_data,
            input=scrub_images_from_items(data.model_data.input, text=_TEXT_ONLY_IMAGE_TEXT),
        )
        if inner is None:
            return model_data
        result = inner(replace(data, model_data=model_data))
        return await result if inspect.isawaitable(result) else result

    return replace(run_config, call_model_input_filter=_scrub)


_MAX_TRANSIENT_MODEL_RETRIES = 5
_TRANSIENT_MODEL_RETRY_BASE_DELAY_S = 2.0
_TRANSIENT_MODEL_RETRY_MAX_DELAY_S = 90.0


def _model_error_status_code(exc: BaseException) -> int | None:
    code = getattr(exc, "status_code", None)
    return code if isinstance(code, int) else None


def _is_image_rejection(exc: BaseException) -> bool:
    return _model_error_status_code(exc) in _INPUT_REJECTION_CODES and bool(
        _IMAGE_REJECTION.search(str(exc))
    )


def _is_transient_model_error(exc: BaseException) -> bool:
    if codex.is_content_guardrail_error(exc):
        return False
    if isinstance(
        exc, APITimeoutError | APIConnectionError | TimeoutError | ConnectionError | OSError
    ):
        return True
    code = _model_error_status_code(exc)
    if code is not None:
        return code >= 400 and code not in (401, 402, 403, 404)
    return isinstance(exc, APIError)


def _transient_model_retry_delay(attempt: int) -> float:
    delay = _TRANSIENT_MODEL_RETRY_BASE_DELAY_S * float(2 ** (attempt - 1))
    return min(delay, _TRANSIENT_MODEL_RETRY_MAX_DELAY_S)


async def _salvage_stream_to_session(
    session: Session,
    pre_run_items: list[Any],
    stream: Any,
    agent_id: str,
) -> None:
    """Persist a crashed run's full history so a revived agent loses no context."""
    if stream is None:
        return
    try:
        replay = list(stream.to_input_list())
    except Exception:
        logger.exception("could not build salvage history for %s", agent_id)
        return
    desired = list(pre_run_items) + replay
    if len(desired) <= len(pre_run_items):
        return
    try:
        await replace_session_items(session, desired)
    except Exception:
        logger.exception("salvaging crashed run history failed for %s", agent_id)


async def _seed_and_prepare_first_input(
    session: Session | None, initial_input: Any, *, start_parked: bool
) -> Any:
    """Persist the opening input up front so it survives a first-turn crash."""
    if initial_input and session is not None and not start_parked:
        with contextlib.suppress(Exception):
            if await seed_initial_input(session, initial_input):
                return []
    return initial_input


async def run_agent_loop(
    *,
    agent: Any,
    initial_input: Any,
    run_config: RunConfig,
    context: dict[str, Any],
    max_turns: int,
    coordinator: AgentCoordinator,
    agent_id: str,
    interactive: bool,
    session: Session | None = None,
    start_parked: bool = False,
    event_sink: StreamEventSink | None = None,
    hooks: RunHooks[dict[str, Any]] | None = None,
) -> RunResultBase | None:
    agent_name = getattr(agent, "name", None)
    token = request_log.bind_call_context(
        agent_id, agent_name if isinstance(agent_name, str) else None
    )
    try:
        return await _run_agent_loop(
            agent=agent,
            initial_input=initial_input,
            run_config=run_config,
            context=context,
            max_turns=max_turns,
            coordinator=coordinator,
            agent_id=agent_id,
            interactive=interactive,
            session=session,
            start_parked=start_parked,
            event_sink=event_sink,
            hooks=hooks,
        )
    finally:
        request_log.reset_call_context(token)


async def _run_agent_loop(
    *,
    agent: Any,
    initial_input: Any,
    run_config: RunConfig,
    context: dict[str, Any],
    max_turns: int,
    coordinator: AgentCoordinator,
    agent_id: str,
    interactive: bool,
    session: Session | None,
    start_parked: bool,
    event_sink: StreamEventSink | None,
    hooks: RunHooks[dict[str, Any]] | None,
) -> RunResultBase | None:
    await coordinator.attach_runtime(
        agent_id,
        session=session,
        interrupt_on_message=interactive,
        resumable=interactive,
    )
    result: RunResultBase | None = None

    first_cycle_input = await _seed_and_prepare_first_input(
        session, initial_input, start_parked=start_parked
    )

    budget_stopped = coordinator.budget_stopped
    reserve_stopped = coordinator.reserve_stopped
    if budget_stopped:
        await coordinator.set_status(agent_id, "stopped")
        raise BudgetExceededError("scan budget reached")
    if reserve_stopped and context.get("parent_id") is not None:
        await coordinator.set_status(agent_id, "stopped")
        raise SubagentBudgetReservedError("scan reached the sub-agent budget reserve")

    if reserve_stopped and start_parked and interactive and context.get("parent_id") is None:
        await coordinator.send(agent_id, _reserve_notice(coordinator.root_finish_tool))

    if not (start_parked and interactive):
        with contextlib.suppress(BudgetPausedError):
            result = await _run_until_lifecycle(
                agent,
                coordinator,
                agent_id,
                initial_input=first_cycle_input,
                run_config=run_config,
                context=context,
                max_turns=max_turns,
                session=session,
                interactive=interactive,
                event_sink=event_sink,
                hooks=hooks,
            )

    if not interactive:
        return result

    while True:
        timeout = await _plain_waiting_timeout(coordinator, agent_id)
        try:
            woke = await coordinator.wait_for_message(agent_id, timeout=timeout)
        except asyncio.CancelledError:
            return result

        if coordinator.budget_stopped:
            await coordinator.set_status(agent_id, "stopped")
            raise BudgetExceededError("scan budget reached")

        if coordinator.reserve_stopped and context.get("parent_id") is not None:
            await coordinator.set_status(agent_id, "stopped")
            raise SubagentBudgetReservedError("scan reached the sub-agent budget reserve")

        if woke:
            # Real input is real progress, so the nudge budget starts over. A bare
            # auto-resume is not: it must not hand a wedged agent a fresh budget.
            await coordinator.reset_recovery(agent_id)
            await coordinator.reset_idle_resumes(agent_id)
        else:
            idle_resumes = await coordinator.record_idle_resume(agent_id)
            if idle_resumes >= _MAX_IDLE_AUTO_RESUMES:
                logger.warning(
                    "agent %s auto-resumed %d times without hearing from anyone; "
                    "leaving it parked until a real message arrives",
                    agent_id,
                    idle_resumes,
                )
                await coordinator.park_waiting(agent_id, wait_kind="stalled")
                await _notify_parent_on_stall(coordinator, agent_id)
                continue
            logger.info("agent %s reached its waiting timeout; auto-resuming", agent_id)
            await coordinator.send(
                agent_id,
                {
                    "from": "system",
                    "type": "auto_resume",
                    "content": "Waiting timeout reached. Resuming execution.",
                },
                interrupt=False,
            )

        await coordinator.consume_pending(agent_id)
        with contextlib.suppress(BudgetPausedError):
            result = await _run_until_lifecycle(
                agent,
                coordinator,
                agent_id,
                initial_input=[],
                run_config=run_config,
                context=context,
                max_turns=max_turns,
                session=session,
                interactive=True,
                event_sink=event_sink,
                hooks=hooks,
            )


async def spawn_child_agent(
    *,
    coordinator: AgentCoordinator,
    factory: Any,
    agents_db_path: Path,
    sessions_to_close: list[SQLiteSession],
    run_config: RunConfig,
    max_turns: int,
    interactive: bool,
    parent_ctx: dict[str, Any],
    name: str,
    task: str,
    skills: list[str],
    parent_history: list[Any],
    event_sink: StreamEventSink | None = None,
    hooks: RunHooks[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    parent_id = parent_ctx.get("agent_id")
    if not isinstance(parent_id, str):
        raise TypeError("Parent agent_id missing from context")

    child_id = uuid.uuid4().hex[:8]
    child_agent = factory(name=name, skills=skills)
    await coordinator.register(
        child_id,
        name,
        parent_id,
        task=task,
        skills=skills,
    )

    await _start_child_runner(
        parent_ctx=parent_ctx,
        coordinator=coordinator,
        agents_db_path=agents_db_path,
        sessions_to_close=sessions_to_close,
        run_config=run_config,
        max_turns=max_turns,
        interactive=interactive,
        child_agent=child_agent,
        child_id=child_id,
        name=name,
        parent_id=parent_id,
        task=task,
        initial_input=child_initial_input(
            name=name,
            child_id=child_id,
            parent_id=parent_id,
            task=task,
            parent_history=parent_history,
        ),
        event_sink=event_sink,
        hooks=hooks,
    )

    return {
        "success": True,
        "agent_id": child_id,
        "name": name,
        "parent_id": parent_id,
        "message": f"Spawned '{name}' ({child_id}) running in parallel.",
    }


async def respawn_subagents(
    *,
    coordinator: AgentCoordinator,
    factory: Any,
    agents_db_path: Path,
    sessions_to_close: list[SQLiteSession],
    run_config: RunConfig,
    max_turns: int,
    interactive: bool,
    parent_ctx: dict[str, Any],
    root_id: str,
    event_sink: StreamEventSink | None = None,
    hooks: RunHooks[dict[str, Any]] | None = None,
) -> None:
    async with coordinator._lock:
        agents_snapshot = [
            (aid, status, dict(coordinator.metadata.get(aid, {})))
            for aid, status in coordinator.statuses.items()
        ]
        candidates: list[tuple[str, str, str | None, dict[str, Any]]] = []
        for aid, status, md in agents_snapshot:
            if not interactive and status not in {"running", "waiting"}:
                continue
            if coordinator.parent_of.get(aid) is None or aid == root_id:
                continue
            md["_restored_status"] = status
            candidates.append(
                (
                    aid,
                    coordinator.names.get(aid, aid),
                    coordinator.parent_of.get(aid),
                    md,
                )
            )

    for child_id, name, parent_id, md in candidates:
        try:
            restored_status = str(md.get("_restored_status") or "running")
            start_parked = interactive and restored_status != "running"

            if start_parked:
                logger.warning(
                    "respawn %s (%s): starting parked from status=%s",
                    child_id,
                    name,
                    restored_status,
                )

            child_skills = list(md.get("skills") or [])
            child_agent = factory(name=name, skills=child_skills)
            await _start_child_runner(
                parent_ctx=parent_ctx,
                coordinator=coordinator,
                agents_db_path=agents_db_path,
                sessions_to_close=sessions_to_close,
                run_config=run_config,
                max_turns=max_turns,
                interactive=interactive,
                child_agent=child_agent,
                child_id=child_id,
                name=name,
                parent_id=parent_id,
                task=str(md.get("task", "")),
                initial_input=[],
                start_parked=start_parked,
                event_sink=event_sink,
                hooks=hooks,
            )
            logger.info(
                "respawned %s (%s) parent=%s task_len=%d",
                child_id,
                name,
                parent_id or "-",
                len(md.get("task", "")),
            )
        except Exception:
            logger.exception("respawn %s failed; marking crashed", child_id)
            with contextlib.suppress(Exception):
                await coordinator.set_status(child_id, "crashed")


_INTERACTIVE_TOOL_RECOVERY_LIMIT = 3


async def _one_run_cycle(
    agent: Any,
    coordinator: AgentCoordinator,
    agent_id: str,
    *,
    input_data: Any,
    run_config: RunConfig,
    context: dict[str, Any],
    max_turns: int,
    session: Session | None,
    interactive: bool,
    event_sink: StreamEventSink | None,
    hooks: RunHooks[dict[str, Any]] | None,
    loop_detector: ThinkingLoopDetector | None = None,
) -> RunResultBase | None:
    """Run one cycle on the path its mode requires.

    Interactive runs park on any error instead of killing the runner; an
    autonomous run's error ends the agent. Keeping the choice here leaves the
    lifecycle loop reading as one sequence of decisions.
    """
    if interactive:
        return await _run_cycle_parked(
            agent,
            coordinator,
            agent_id,
            input_data=input_data,
            run_config=run_config,
            context=context,
            max_turns=max_turns,
            session=session,
            event_sink=event_sink,
            hooks=hooks,
            loop_detector=loop_detector,
        )
    return await _run_cycle(
        agent,
        coordinator,
        agent_id,
        input_data=input_data,
        run_config=run_config,
        context=context,
        max_turns=max_turns,
        session=session,
        interactive=False,
        event_sink=event_sink,
        hooks=hooks,
        loop_detector=loop_detector,
    )


async def _run_until_lifecycle(
    agent: Any,
    coordinator: AgentCoordinator,
    agent_id: str,
    *,
    initial_input: Any,
    run_config: RunConfig,
    context: dict[str, Any],
    max_turns: int,
    session: Session | None,
    interactive: bool,
    event_sink: StreamEventSink | None,
    hooks: RunHooks[dict[str, Any]] | None,
) -> RunResultBase | None:
    """Drive an agent until an explicit lifecycle tool settles its status.

    A turn that ends without ``finish_scan``, ``agent_finish``,
    ``wait_for_user``, or ``wait_for_agents`` leaves the agent ``running``:
    plain text never terminates a run and never yields to the user. Such a turn
    is nudged back into a tool call, bounded by a recovery limit. The same
    budget covers a ``wait_for_user`` call made before anything was said to the
    user since their last message: plain text is the only channel to them, so
    that park would hand them a silent turn, and the agent is sent back to
    write its reply instead.
    """
    result: RunResultBase | None = None
    input_data: Any = initial_input
    recovery_limit = _INTERACTIVE_TOOL_RECOVERY_LIMIT if interactive else max(1, max_turns)
    said_to_user = False
    detector = ThinkingLoopDetector()
    recovery = LoopRecoveryController()
    replacements = 0

    while True:
        if coordinator.budget_stopped:
            await coordinator.set_status(agent_id, "stopped")
            raise BudgetExceededError("scan budget reached")

        if coordinator.reserve_stopped and context.get("parent_id") is not None:
            await coordinator.set_status(agent_id, "stopped")
            raise SubagentBudgetReservedError("scan reached the sub-agent budget reserve")

        try:
            result = await _one_run_cycle(
                agent,
                coordinator,
                agent_id,
                input_data=input_data,
                run_config=run_config,
                context=context,
                max_turns=max_turns,
                session=session,
                interactive=interactive,
                event_sink=event_sink,
                hooks=hooks,
                loop_detector=detector,
            )
        except ThinkingLoopError as exc:
            detector.reset()
            if replacements >= _MAX_SESSION_REPLACEMENTS:
                logger.warning(
                    "agent %s stalled in a thinking loop again after %d session "
                    "replacement(s); settling it instead of replacing forever",
                    agent_id,
                    replacements,
                )
                return await _exhausted_recovery(
                    coordinator, agent_id, None, interactive=interactive
                )
            outcome = recovery.on_loop(exc.signal)
            logger.warning(
                "agent %s loop recovery step '%s' (%s)",
                agent_id,
                outcome.action,
                exc.signal.describe(),
            )
            session, replacements = await _recover_from_loop(
                outcome=outcome,
                agent=agent,
                coordinator=coordinator,
                agent_id=agent_id,
                session=session,
                run_config=run_config,
                interactive=interactive,
                replacements=replacements,
            )
            input_data = []
            continue
        except BudgetPausedError as exc:
            if coordinator.budget_policy != "pause":
                raise
            # The agent parked right before an LLM call; everything up to that
            # point is already in its session. Once resumed, the same call goes
            # out with nothing added to the conversation.
            await coordinator.wait_for_budget_resume(agent_id, parked_epoch=exc.resume_epoch)
            if (
                not coordinator.budget_stopped
                and await _agent_status(coordinator, agent_id) != "running"
            ):
                # Stopped while parked (operator stop or a parent's stop_agent).
                await coordinator.reset_recovery(agent_id)
                return result
            if session is not None:
                input_data = []
            continue

        said_to_user = said_to_user or _said_to_user(result)
        # Atomic: only an agent still parked on the user is put back to work, so
        # a stop that lands in between is never overwritten.
        silent_yield = (
            interactive and not said_to_user and await coordinator.resume_silent_user_wait(agent_id)
        )
        status = await _agent_status(coordinator, agent_id)
        if status != "running" and not silent_yield:
            detector.reset()
            recovery.note_progress()
            await coordinator.reset_recovery(agent_id)
            return result

        recoveries = await coordinator.record_recovery(agent_id)
        _log_recovery(
            agent_id,
            result,
            recoveries,
            recovery_limit,
            interactive=interactive,
            silent_yield=silent_yield,
        )
        if recoveries >= recovery_limit:
            return await _exhausted_recovery(coordinator, agent_id, result, interactive=interactive)

        input_data = await _append_tool_required_message(
            session=session,
            context=context,
            root_finish_tool=coordinator.root_finish_tool,
            attempt=recoveries,
            limit=recovery_limit,
            interactive=interactive,
            silent_yield=silent_yield,
        )


async def _recover_from_loop(
    *,
    outcome: Any,
    agent: Any,
    coordinator: AgentCoordinator,
    agent_id: str,
    session: Session | None,
    run_config: RunConfig,
    interactive: bool,
    replacements: int,
) -> tuple[Session | None, int]:
    """Run one ladder step, returning the (possibly replaced) session.

    The replacement session is opened before the old one is dropped and is
    attached to the coordinator, so a message queued mid-recovery lands in the
    session the agent will actually read next. The prior session's history is
    carried across verbatim: compaction/session replacement must never lose
    findings or scan state.

    Interactive runs read their mailbox between cycles, so the continuation
    prompt is queued there. An autonomous run's loop never drains a mailbox, so
    the prompt is appended straight to its session.
    """

    async def _send_prompt(prompt: str) -> bool:
        if interactive:
            return await coordinator.send(
                agent_id,
                {
                    "from": "system",
                    "type": "loop_recovery",
                    "priority": "high",
                    "content": prompt,
                },
                interrupt=False,
            )
        if session is None:
            return False
        try:
            await session.add_items(
                [cast("TResponseInputItem", {"role": "user", "content": prompt})]
            )
        except Exception:
            logger.exception("appending the loop-recovery prompt failed for %s", agent_id)
            return False
        return True

    async def _compact() -> bool:
        if session is None:
            return False
        try:
            return await _compact_session(agent, session, run_config, force=True)
        except Exception:
            logger.exception("forced compaction during loop recovery failed for %s", agent_id)
            return False

    async def _replace() -> Session | None:
        if session is None:
            return None
        new_session = open_replacement_session(session)
        if new_session is None:
            logger.warning(
                "agent %s cannot open a replacement session (no on-disk session db); "
                "continuing in the current session",
                agent_id,
            )
            return None
        try:
            items = list(await session.get_items())
        except Exception:
            logger.exception("reading the prior session during replacement failed for %s", agent_id)
            items = []
        if items:
            with contextlib.suppress(Exception):
                await new_session.add_items(items)
        await coordinator.attach_runtime(agent_id, session=new_session)
        # ``Session`` has no ``close`` on the protocol; the concrete SQLite
        # session does, and leaving it open would leak a file handle per
        # replacement, so it is closed defensively.
        with contextlib.suppress(Exception):
            close = getattr(session, "close", None)
            if callable(close):
                close()
        logger.warning(
            "agent %s replaced session %s with %s carrying %d item(s)",
            agent_id,
            getattr(session, "session_id", "?"),
            getattr(new_session, "session_id", "?"),
            len(items),
        )
        return new_session

    adapter = RecoveryAdapter(
        cancel_stream=lambda: None,
        send_prompt=_send_prompt,
        compact=_compact,
        replace_session=_replace,
    )
    new_session = await apply_recovery(outcome, adapter)
    if new_session is not None:
        return new_session, replacements + 1
    return session, replacements


async def _exhausted_recovery(
    coordinator: AgentCoordinator,
    agent_id: str,
    result: RunResultBase | None,
    *,
    interactive: bool,
) -> RunResultBase | None:
    """Settle an agent that never recovered into a tool call.

    Interactive runs park instead of dying: a human is attached and can message
    any agent, so the scan stays resumable. Autonomous runs have nobody to
    resume them, so they fail loudly.
    """
    if not interactive:
        await coordinator.set_status(agent_id, "crashed")
        await notify_parent_on_terminal(coordinator, agent_id, "crashed")
        raise MaxTurnsExceeded(
            "Agent exhausted recovery attempts without calling finish_scan or agent_finish."
        )

    logger.warning(
        "agent %s exhausted tool-call recovery attempts; parking until a message arrives",
        agent_id,
    )
    await coordinator.park_waiting(agent_id, wait_kind="stalled")
    # A parked child owes its parent a completion report it can no longer send. The
    # parent is an agent, not a watching human, so nothing else tells it to stop
    # waiting and it burns its full timeout on a message that is never coming.
    await _notify_parent_on_stall(coordinator, agent_id)
    return result


_WAITING_AUTO_RESUME_TIMEOUT_S = 300.0

# An agent that parks again after every auto-resume makes no progress, so stop
# spending a model turn per timeout and leave it parked for a real message.
_MAX_IDLE_AUTO_RESUMES = 3


async def _plain_waiting_timeout(
    coordinator: AgentCoordinator,
    agent_id: str,
) -> float | None:
    """Auto-resume timeout for a parked agent; None waits until a message arrives.

    Driven by what the agent is waiting on, not by where it sits in the graph:
    the user can message any agent, so an agent awaiting a human parks
    indefinitely whether or not it is the root. Only an agent awaiting other
    agents is re-checked on a timer, and only until it has spent its idle
    budget re-parking without hearing anything.
    """
    async with coordinator._lock:
        status = coordinator.statuses.get(agent_id)
        has_error = agent_id in coordinator.errors
        runtime = coordinator.runtimes.get(agent_id)
        gated = runtime.user_wake_required if runtime is not None else False
        wait_kind = coordinator.wait_kinds.get(agent_id)
        idle_resumes = coordinator.idle_resume_counts.get(agent_id, 0)
    if status != "waiting" or has_error or gated:
        return None
    if wait_kind != "agents" or idle_resumes >= _MAX_IDLE_AUTO_RESUMES:
        return None
    return _WAITING_AUTO_RESUME_TIMEOUT_S


async def _run_cycle_parked(
    agent: Any,
    coordinator: AgentCoordinator,
    agent_id: str,
    *,
    input_data: Any,
    run_config: RunConfig,
    context: dict[str, Any],
    max_turns: int,
    session: Session | None,
    event_sink: StreamEventSink | None,
    hooks: RunHooks[dict[str, Any]] | None,
    loop_detector: ThinkingLoopDetector | None = None,
) -> RunResultBase | None:
    """Interactive run cycle that parks on any error instead of killing the runner."""
    try:
        return await _run_cycle(
            agent,
            coordinator,
            agent_id,
            input_data=input_data,
            run_config=run_config,
            context=context,
            max_turns=max_turns,
            session=session,
            interactive=True,
            event_sink=event_sink,
            hooks=hooks,
            loop_detector=loop_detector,
        )
    except (BudgetExceededError, BudgetPausedError, SubagentBudgetReservedError):
        raise
    except ThinkingLoopError:
        # Loop recovery is a control signal, not a failure: hand it to the
        # ladder in ``_run_until_lifecycle`` instead of parking the agent.
        raise
    except Exception as exc:
        logger.exception("error escaped the run cycle for %s; parking as failed", agent_id)
        await coordinator.set_status(agent_id, "failed", error=request_log.failure_text(exc))
        await notify_parent_on_terminal(coordinator, agent_id, "failed")
        return None


async def _run_cycle(  # noqa: PLR0912, PLR0915
    agent: Any,
    coordinator: AgentCoordinator,
    agent_id: str,
    *,
    input_data: Any,
    run_config: RunConfig,
    context: dict[str, Any],
    max_turns: int,
    session: Session | None,
    interactive: bool,
    event_sink: StreamEventSink | None,
    hooks: RunHooks[dict[str, Any]] | None,
    loop_detector: ThinkingLoopDetector | None = None,
) -> RunResultBase | None:
    image_strips = 0
    compactions = 0
    model_retries = 0
    request_log.set_retry_attempt(0)
    while True:
        stream: Any = None
        pre_run_items: list[Any] = []
        try:
            await coordinator.mark_running(agent_id)
            if session is not None:
                max_images = context.get("max_context_images")
                if isinstance(max_images, int):
                    try:
                        await enforce_image_budget(session, max_images)
                    except Exception:
                        logger.exception("image-budget enforcement failed for %s", agent_id)
                try:
                    await _compact_session(agent, session, run_config, force=False)
                except Exception:
                    logger.exception("proactive compaction failed for %s", agent_id)
                with contextlib.suppress(Exception):
                    pre_run_items = list(await session.get_items())
            stream = Runner.run_streamed(
                agent,
                input=input_data,
                run_config=_with_image_scrub(run_config, context),
                context=context,
                max_turns=max_turns,
                session=session,
                hooks=hooks,
            )
            await coordinator.attach_stream(agent_id, stream)
            try:
                try:
                    async for event in _run_stream_events(stream, loop_detector):
                        if event_sink is not None:
                            try:
                                event_sink(agent_id, event)
                            except Exception:
                                logger.exception("stream event sink failed for %s", agent_id)
                    if stream.run_loop_exception is not None:
                        raise stream.run_loop_exception
                    if refusal := _structured_provider_refusal(stream):
                        raise ProviderRefusalError(refusal)
                except ThinkingLoopError:
                    raise
                except (BudgetExceededError, BudgetPausedError, SubagentBudgetReservedError):
                    raise
                except RuntimeError as stream_exc:
                    if "after shutdown" not in str(stream_exc):
                        raise
                    logger.warning(
                        "Ignoring LiteLLM end-of-stream shutdown race for %s",
                        agent_id,
                    )
                except _teardown_sandbox_errors():
                    if not coordinator.is_shutting_down:
                        raise
                    logger.warning(
                        "Ignoring sandbox container error during teardown for %s",
                        agent_id,
                        exc_info=True,
                    )
            finally:
                await coordinator.detach_stream(agent_id, stream)
        except BudgetPausedError as exc:
            logger.info("agent %s paused at the scan budget limit: %s", agent_id, exc)
            if coordinator.budget_policy == "pause":
                await coordinator.park_for_budget(agent_id)
            else:
                await coordinator.pause_for_budget(agent_id)
            raise
        except SubagentBudgetReservedError as exc:
            logger.info("sub-agent %s stopped at the budget reserve: %s", agent_id, exc)
            await coordinator.set_status(agent_id, "stopped")
            await _notify_root_on_budget_reserve(coordinator)
            raise
        except BudgetExceededError as exc:
            logger.info(
                "agent %s reached the scan budget limit; stopping the scan: %s", agent_id, exc
            )
            await coordinator.set_status(agent_id, "stopped")
            await coordinator.trigger_budget_stop()
            raise
        except ThinkingLoopError as exc:
            # The stalled turn's history is salvaged before the ladder runs, so
            # the interrupt loses nothing the model already produced.
            if session is not None:
                await _salvage_stream_to_session(session, pre_run_items, stream, agent_id)
            logger.warning("agent %s stalled in a thinking loop: %s", agent_id, exc)
            raise
        except Exception as exc:
            if image_strips < 3 and session is not None and _is_image_rejection(exc):
                try:
                    stripped = await strip_all_images_from_session(session)
                except Exception:
                    logger.exception("image-strip recovery failed for %s", agent_id)
                    stripped = False
                if stripped:
                    image_strips += 1
                    logger.info(
                        "Stripped images from %s session after rejection; retrying (%d)",
                        agent_id,
                        image_strips,
                    )
                    input_data = []
                    continue
            if (
                compactions < _MAX_COMPACTIONS_PER_CYCLE
                and session is not None
                and is_context_overflow(exc)
            ):
                try:
                    compacted = await _compact_session(agent, session, run_config, force=True)
                except Exception:
                    logger.exception("overflow compaction recovery failed for %s", agent_id)
                    compacted = False
                if compacted:
                    compactions += 1
                    logger.info(
                        "Compacted %s session after context overflow; retrying (%d)",
                        agent_id,
                        compactions,
                    )
                    input_data = []
                    continue
            if model_retries < _MAX_TRANSIENT_MODEL_RETRIES and _is_transient_model_error(exc):
                model_retries += 1
                delay = _transient_model_retry_delay(model_retries)
                logger.warning(
                    "transient model/provider error for %s; replaying turn "
                    "(attempt %d/%d, backoff %.1fs): %r",
                    agent_id,
                    model_retries,
                    _MAX_TRANSIENT_MODEL_RETRIES,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)
                request_log.set_retry_attempt(model_retries)
                if session is not None:
                    input_data = []
                continue
            if session is not None:
                await _salvage_stream_to_session(session, pre_run_items, stream, agent_id)
            if isinstance(exc, ProviderRefusalError):
                logger.warning("agent %s refused by the model provider: %s", agent_id, exc)
                await coordinator.set_status(
                    agent_id, "failed", error=request_log.failure_text(exc)
                )
                await notify_parent_on_terminal(coordinator, agent_id, "failed")
                return None
            if isinstance(exc, MaxTurnsExceeded):
                status: Status = "stopped"
            elif isinstance(exc, UserError | AgentsException | APIError):
                status = "failed"
            else:
                status = "crashed"
            logger.exception("agent run failed for %s; marking %s", agent_id, status)
            # Settle the status and wake the parent before the exception unwinds a
            # non-interactive agent's task: a child that dies still owes its parent a
            # report, and the parent would otherwise wait out its timeout on a message
            # the dead child can no longer send.
            await coordinator.set_status(agent_id, status, error=request_log.failure_text(exc))
            await notify_parent_on_terminal(coordinator, agent_id, status)
            if not interactive:
                raise
            return None
        else:
            return cast("RunResultBase | None", stream)


async def _agent_status(coordinator: AgentCoordinator, agent_id: str) -> Status | None:
    async with coordinator._lock:
        return coordinator.statuses.get(agent_id)


def _log_recovery(
    agent_id: str,
    result: RunResultBase | None,
    attempt: int,
    limit: int,
    *,
    interactive: bool,
    silent_yield: bool,
) -> None:
    if silent_yield:
        logger.warning(
            "agent %s called wait_for_user without saying anything to the user; "
            "sending it back to reply (%d/%d)",
            agent_id,
            attempt,
            limit,
        )
        return
    logger.warning(
        "agent %s ended a turn without a lifecycle tool call (interactive=%s); "
        "forcing tool continuation (%d/%d): %s",
        agent_id,
        interactive,
        attempt,
        limit,
        _final_output_preview(result),
    )


def _said_to_user(result: RunResultBase | None) -> bool:
    """Whether the run produced any assistant text, the only channel to the user."""
    for item in getattr(result, "new_items", ()) or ():
        if isinstance(item, MessageOutputItem) and ItemHelpers.text_message_output(item).strip():
            return True
    return False


def _final_output_preview(result: RunResultBase | None) -> str:
    final_output = getattr(result, "final_output", None)
    if final_output is None:
        return "<none>"
    text = str(final_output).replace("\n", " ").strip()
    if not text:
        return "<empty>"
    return text[:300]


async def _append_tool_required_message(
    *,
    session: Session | None,
    context: dict[str, Any],
    attempt: int,
    limit: int,
    interactive: bool,
    silent_yield: bool = False,
    root_finish_tool: str = "finish_scan",
) -> list[dict[str, str]]:
    finish_tool = root_finish_tool if context.get("parent_id") is None else "agent_finish"
    if silent_yield:
        message = (
            "You called wait_for_user without having written anything to the user since "
            "their last message, so they would be handed a silent turn. Plain text is the "
            "only channel to the user: write your reply as plain text now, then call "
            f"wait_for_user. This is recovery attempt {attempt}/{limit}."
        )
    elif interactive:
        message = (
            "Your previous message ended a turn without a tool call. Plain text never ends "
            "execution and never hands control to the user: it is shown to the user, and the "
            "run continues. Continue immediately and call exactly one tool. "
            "If you have nothing to do until the user replies, call wait_for_user; your "
            "text already reached them, so do not repeat it. "
            "If you are blocked waiting for another agent, call wait_for_agents. "
            f"If the whole engagement is complete, call {finish_tool}. "
            "Otherwise use the appropriate execution or planning tool. "
            f"This is recovery attempt {attempt}/{limit}."
        )
    else:
        message = (
            "Your previous response ended the autonomous run without a lifecycle tool "
            "call. That is invalid in non-interactive mode; plain text final answers are "
            "ignored. Continue immediately and call exactly one tool. "
            f"If your work is complete, call {finish_tool}. "
            "If you are blocked waiting for another agent, call wait_for_agents. "
            "Otherwise use the appropriate execution or planning tool. "
            f"This is recovery attempt {attempt}/{limit}."
        )
    item = {"role": "user", "content": message}
    if session is None:
        return [item]

    await session.add_items([cast("TResponseInputItem", item)])
    return []


_TERMINAL_NOTICE = {
    "completed": (
        "[Agent completed] {name} ({agent_id}) finished and is no longer running, but it "
        "sent no completion report. Stop waiting on this child; ask it directly if you "
        "need its results."
    ),
    "crashed": (
        "[Agent crash] {name} ({agent_id}) terminated unexpectedly. "
        "Stop waiting on this child unless you want to message it again."
    ),
    "failed": (
        "[Agent failed] {name} ({agent_id}) stopped with an error and will not "
        "send a completion report. Stop waiting on this child unless you want to "
        "message it again."
    ),
    "stopped": (
        "[Agent stopped] {name} ({agent_id}) was stopped before finishing (turn limit "
        "or an explicit stop). It will not send a completion report, so stop waiting "
        "on this child; account for its unfinished subtask and continue."
    ),
}


_STALL_NOTICE = (
    "[Agent stalled] {name} ({agent_id}) kept ending turns without a tool call and is "
    "parked until it receives a message. It will not send a completion report on its "
    "own: either message it with a concrete next step to unblock it, or stop waiting on "
    "it and account for its unfinished subtask."
)


async def _notify_parent_on_stall(
    coordinator: AgentCoordinator,
    agent_id: str,
) -> None:
    """Tell the parent that a child parked mid-task, so it stops waiting blindly."""
    async with coordinator._lock:
        parent = coordinator.parent_of.get(agent_id)
        name = coordinator.names.get(agent_id, agent_id)
    if parent is None:
        return
    await coordinator.send(
        parent,
        {
            "from": agent_id,
            "type": "stalled",
            "priority": "high",
            "content": _STALL_NOTICE.format(name=name, agent_id=agent_id),
        },
        interrupt=False,
    )


async def notify_parent_on_terminal(
    coordinator: AgentCoordinator,
    agent_id: str,
    status: str,
) -> None:
    template = _TERMINAL_NOTICE.get(status)
    if template is None:
        return
    async with coordinator._lock:
        parent = coordinator.parent_of.get(agent_id)
        name = coordinator.names.get(agent_id, agent_id)
    if parent is None:
        return
    if not await coordinator.claim_parent_notice(agent_id):
        return
    await coordinator.send(
        parent,
        {
            "from": agent_id,
            "type": status,
            "priority": "high",
            "content": template.format(name=name, agent_id=agent_id),
        },
        interrupt=False,
    )


def _reserve_notice(finish_tool: str) -> dict[str, Any]:
    return {
        "from": "system",
        "type": "budget_reserve_stop",
        "priority": "high",
        "content": (
            "[Budget reserve] The scan has reached the sub-agent budget reserve: every "
            "sub-agent is being force-stopped as soon as its in-flight turn completes, and "
            "none will send a completion report. Their confirmed vulnerabilities are "
            "already filed as they were found. Do not wait on any sub-agents and do not "
            f"spawn new ones — wrap up now and call {finish_tool}."
        ),
    }


async def _notify_root_on_budget_reserve(coordinator: AgentCoordinator) -> None:
    root = await coordinator.claim_reserve_notification()
    if root is None:
        return
    await coordinator.send(root, _reserve_notice(coordinator.root_finish_tool))


async def _notify_parent_on_exit(
    coordinator: AgentCoordinator,
    agent_id: str,
) -> None:
    """Backstop for a child whose loop ended without telling its parent.

    Every terminal state counts, including ``completed``: a child that skips its
    completion report leaves the parent waiting on a message nobody will send.
    """
    status = await _agent_status(coordinator, agent_id)
    if status is None:
        return
    await notify_parent_on_terminal(coordinator, agent_id, status)


async def _start_child_runner(
    *,
    parent_ctx: dict[str, Any],
    coordinator: AgentCoordinator,
    agents_db_path: Path,
    sessions_to_close: list[SQLiteSession],
    run_config: RunConfig,
    max_turns: int,
    interactive: bool,
    child_agent: Any,
    child_id: str,
    name: str,
    parent_id: str | None,
    task: str,
    initial_input: Any,
    start_parked: bool = False,
    event_sink: StreamEventSink | None = None,
    hooks: RunHooks[dict[str, Any]] | None = None,
) -> None:
    session = open_agent_session(child_id, agents_db_path)
    sessions_to_close.append(session)
    await coordinator.attach_runtime(child_id, session=session, resumable=interactive)

    child_ctx: dict[str, Any] = dict(parent_ctx)
    child_ctx["agent_id"] = child_id
    child_ctx["parent_id"] = parent_id
    child_ctx["task"] = task

    async def _child_loop() -> None:
        # A budget stop is a clean scan-wide shutdown, not a child failure: the
        # child's status and parent notification are already settled in
        # ``_run_cycle``. Swallow it here so the detached task does not surface a
        # spurious "Task exception was never retrieved" warning. The root agent
        # hits the same limit on its next call and tears the scan down.
        try:
            await run_agent_loop(
                agent=child_agent,
                initial_input=initial_input,
                run_config=run_config,
                context=child_ctx,
                max_turns=max_turns,
                coordinator=coordinator,
                agent_id=child_id,
                interactive=interactive,
                session=session,
                start_parked=start_parked,
                event_sink=event_sink,
                hooks=hooks,
            )
        except BudgetExceededError:
            logger.info("child %s stopped after reaching the scan budget limit", child_id)
        except SubagentBudgetReservedError:
            logger.info("child %s stopped at the sub-agent budget reserve", child_id)
        finally:
            if not coordinator.is_shutting_down:
                await _notify_parent_on_exit(coordinator, child_id)

    task_handle = asyncio.create_task(_child_loop(), name=f"agent-{name}-{child_id}")
    await coordinator.attach_runtime(child_id, task=task_handle)
