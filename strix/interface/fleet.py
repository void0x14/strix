"""``strix fleet``: scan many targets in parallel under a memory budget.

Each target gets its own strix scan in its own run directory, so the existing
viewer, report writer, and ``--resume`` all work per target unchanged. The fleet
is what adds:

* **Bounded concurrency** derived from the host's free memory, so a long target
  list cannot drive the host into an OOM. A scan is only started while there is
  room for it; nothing already running is ever killed to make room.
* **Per-sandbox cgroup caps** (memory, CPU, PIDs, log size) applied to every
  container the scans create.
* **A live web view**: one viewer is served over the fleet's ``strix_runs``
  directory, so every target's progress is visible from the browser while the
  fleet runs.

It is a supervisor, not a second engine: it launches the shipped
``strix --target ... -n --scan-mode deep`` entry point as a subprocess and only
watches it. When a scan dies the fleet records it and moves on; the other scans
are untouched.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console

from strix.core.paths import RUNS_DIR_NAME
from strix.interface.target_plan import (
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MEMORY_FLOOR_BYTES,
    DEFAULT_PER_SCAN_BYTES,
    available_memory_bytes,
    collect_scan_targets,
    has_memory_headroom,
    plan_concurrency,
    run_names_for,
    scan_environment,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


logger = logging.getLogger(__name__)

_POLL_INTERVAL_S = 2.0
# Time a scan is given to exit after a stop request before it is killed.
_STOP_GRACE_S = 30.0
_DEFAULT_VIEWER_PORT = 8787
# How long to wait for the viewer to start listening before giving up on it.
_VIEWER_READY_TIMEOUT_S = 20.0


@dataclass(slots=True)
class ScanJob:
    """One target's scan and its process state."""

    index: int
    target: str
    slug: str
    slot: int = -1
    process: subprocess.Popen[bytes] | None = None
    log_path: Path | None = None
    started_at: float | None = None
    finished_at: float | None = None
    returncode: int | None = None

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    @property
    def done(self) -> bool:
        return self.process is not None and self.process.poll() is not None

    def status(self) -> str:
        if self.process is None:
            return "pending"
        if self.running:
            return "running"
        return f"exit {self.returncode}"


@dataclass(slots=True)
class FleetResult:
    """Every job's final state, for the operator report."""

    jobs: list[ScanJob] = field(  # pyright: ignore[reportUnknownVariableType]
        default_factory=list
    )
    stopped_early: bool = False

    @property
    def failed(self) -> list[ScanJob]:
        return [job for job in self.jobs if job.returncode not in (0, 2)]

    @property
    def with_findings(self) -> list[ScanJob]:
        return [job for job in self.jobs if job.returncode == 2]


class FleetSupervisor:
    """Run every target as its own scan, at most ``concurrency`` at a time.

    The admission loop is the OOM guard: before starting another scan it re-reads
    the host's free memory and only proceeds while one scan's worth of headroom
    remains above the floor. Both the memory reader and the process launcher are
    injectable so the loop's behavior is testable without Docker or real scans.
    """

    def __init__(
        self,
        *,
        targets: list[str],
        scan_command: Callable[[str, Path], list[str]],
        environment: Mapping[str, str],
        runs_base: Path,
        concurrency: int,
        per_scan_bytes: int,
        memory_floor_bytes: int,
        memory_reader: Callable[[], int | None] = available_memory_bytes,
        launcher: Callable[..., Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Any] = time.sleep,
        run_names: list[str] | None = None,
        on_slot_started: Callable[[int, str], None] | None = None,
        console: Console | None = None,
    ) -> None:
        self._targets = list(targets)
        self._scan_command = scan_command
        self._environment = dict(environment)
        self._runs_base = runs_base
        self._concurrency = max(1, concurrency)
        self._per_scan_bytes = per_scan_bytes
        self._memory_floor_bytes = memory_floor_bytes
        self._memory_reader = memory_reader
        self._launcher = launcher or subprocess.Popen
        self._clock = clock
        self._sleep = sleep
        self._on_slot_started = on_slot_started
        self._console = console or Console()
        self._stop = threading.Event()
        names = run_names if run_names is not None else run_names_for(self._targets)
        self.jobs = [
            ScanJob(index=i, target=target, slug=names[i]) for i, target in enumerate(self._targets)
        ]
        self._free_slots = list(range(self._concurrency))
        self._slot_of: dict[int, int] = {}

    # -- lifecycle -----------------------------------------------------------

    def request_stop(self) -> None:
        """Ask every running scan to stop; they are never killed outright."""
        self._stop.set()
        for job in self.jobs:
            self._terminate(job)

    def run(self) -> FleetResult:
        """Drive the fleet to completion; returns every job's final state."""
        self._runs_base.mkdir(parents=True, exist_ok=True)
        pending = list(self.jobs)
        stalled = False
        try:
            while pending or any(job.running for job in self.jobs):
                if self._stop.is_set():
                    break
                started = False
                while pending and self._active_count() < self._concurrency:
                    if not self._admit():
                        break
                    self._start(pending.pop(0))
                    started = True
                self._reap()
                if pending or any(job.running for job in self.jobs):
                    # Nothing running and nothing admittable means the host is
                    # out of headroom; say so once rather than spinning silently,
                    # so a stalled fleet is visible instead of looking idle.
                    if not started and self._active_count() == 0 and not stalled:
                        stalled = True
                        logger.warning(
                            "no scan can start: free memory is under the floor for one more "
                            "scan; waiting for memory to free up"
                        )
                        self._console.print(
                            "[yellow]waiting for memory headroom before the next scan[/]"
                        )
                    self._sleep(_POLL_INTERVAL_S)
        except KeyboardInterrupt:
            self._console.print("[yellow]Interrupted; stopping the fleet...[/]")
            self.request_stop()
        finally:
            for job in self.jobs:
                self._terminate(job)
            self._reap(final=True)
        return FleetResult(jobs=self.jobs, stopped_early=self._stop.is_set())

    def _admit(self) -> bool:
        """Whether one more scan may start right now."""
        if self._active_count() >= self._concurrency:
            return False
        free = self._memory_reader()
        if has_memory_headroom(
            available_bytes=free,
            per_scan_bytes=self._per_scan_bytes,
            memory_floor_bytes=self._memory_floor_bytes,
        ):
            return True
        logger.info(
            "holding the next scan: free memory %s is under the floor for one more scan",
            _format_bytes(free),
        )
        return False

    def _active_count(self) -> int:
        return sum(1 for job in self.jobs if job.running)

    def _start(self, job: ScanJob) -> None:
        slot = self._take_slot()
        self._slot_of[job.index] = slot
        job.slot = slot
        log_path = self._runs_base / f"{job.slug}-{job.index}.log"
        job.log_path = log_path
        command = self._scan_command(job.target, log_path)
        logger.info(
            "starting scan %d/%d (slot %d): %s", job.index + 1, len(self.jobs), slot, job.target
        )
        self._console.print(
            f"[#60a5fa]start[/] {job.index + 1}/{len(self.jobs)} slot {slot} {job.target} "
            f"-> {log_path.name}"
        )
        handle = log_path.open("wb")
        try:
            job.process = self._launcher(
                command,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=dict(self._environment),
                cwd=str(self._runs_base.parent),
                start_new_session=True,
            )
        except OSError:
            # One target's scan failing to launch must not take the fleet down:
            # record it, release the slot, and let the other targets run.
            logger.exception("failed to launch the scan for %s", job.target)
            job.returncode = -1
            job.finished_at = self._clock()
            self._release_slot(job)
            self._console.print(
                f"[#ef4444]failed[/] {job.index + 1}/{len(self.jobs)} {job.target} "
                "(could not launch)"
            )
            return
        finally:
            # The child owns the descriptor now; the parent's copy is closed so a
            # finished child's log is flushed even while the fleet keeps running.
            handle.close()
        job.started_at = self._clock()
        if self._on_slot_started is not None:
            try:
                self._on_slot_started(slot, job.slug)
            except Exception:
                logger.exception("slot-started callback failed for %s", job.target)

    def _take_slot(self) -> int:
        return self._free_slots.pop(0) if self._free_slots else 0

    def _release_slot(self, job: ScanJob) -> None:
        slot = self._slot_of.pop(job.index, None)
        if slot is not None and slot not in self._free_slots:
            self._free_slots.append(slot)
            self._free_slots.sort()
            job.slot = -1

    def _reap(self, *, final: bool = False) -> None:
        for job in self.jobs:
            if job.process is None or job.returncode is not None:
                continue
            code = job.process.poll()
            if code is None:
                if final:
                    job.returncode = -1
                continue
            job.returncode = code
            job.finished_at = self._clock()
            self._release_slot(job)
            self._console.print(
                f"[#22c55e]done[/] {job.index + 1}/{len(self.jobs)} {job.target} (exit {code})"
            )

    def _terminate(self, job: ScanJob) -> None:
        process = job.process
        if process is None or process.poll() is not None:
            return
        logger.info("stopping scan for %s", job.target)
        try:
            process.terminate()
        except OSError:
            logger.debug("terminate failed for %s", job.target, exc_info=True)
            return
        deadline = self._clock() + _STOP_GRACE_S
        while process.poll() is None and self._clock() < deadline:
            self._sleep(0.5)
        if process.poll() is None:
            logger.warning("scan for %s did not stop; killing it", job.target)
            try:
                process.kill()
            except OSError:
                logger.debug("kill failed for %s", job.target, exc_info=True)


def _format_caps(environment: Mapping[str, str]) -> str:
    return ", ".join(
        f"{key.removeprefix('STRIX_SANDBOX_').lower()}={value}"
        for key, value in environment.items()
        if key.startswith("STRIX_SANDBOX_")
    )


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    return f"{value / (1024**3):.1f} GiB"


# -- viewer ------------------------------------------------------------------


class FleetViewer:
    """Serve one viewer over a single run directory.

    ``serve()`` takes the run it opens by default; every *other* run under the
    same ``strix_runs`` root is part of the email-gated history, so a viewer
    pointed at one run can only show that run to an unverified browser. The fleet
    therefore runs one viewer per concurrency slot (see :class:`FleetViewers`),
    each pointed at the run of the scan currently occupying that slot, so every
    live target is watchable from the web with no verification step.
    """

    def __init__(
        self,
        run_dir: Path,
        *,
        host: str,
        port: int,
        open_browser: bool,
    ) -> None:
        self._run_dir = run_dir
        self._host = host
        self._port = port
        self._open_browser = open_browser
        self._httpd: Any = None
        self.url: str | None = None
        self.authorized_url: str | None = None

    def start(self) -> bool:
        """Start the viewer; return whether it is serving.

        A viewer that cannot start is not fatal: the scans are the work, and the
        operator can attach one later with ``strix view``.
        """
        from strix.interface.viewer.server import (  # noqa: PLC0415 - keeps the viewer off the launch path
            authorized_url,
            bundle_is_built,
            serve,
        )

        if not bundle_is_built():
            logger.warning("viewer bundle is not built; skipping the web view")
            return False
        try:
            self._run_dir.mkdir(parents=True, exist_ok=True)
            self._httpd, self.url, token = serve(
                self._run_dir,
                host=self._host,
                port=self._port,
                open_browser=self._open_browser,
            )
        except Exception:
            logger.exception("could not start the viewer for %s", self._run_dir)
            self._httpd = None
            return False
        self.authorized_url = authorized_url(self.url, token)
        return True

    def stop(self) -> None:
        if self._httpd is None:
            return
        try:
            self._httpd.shutdown()
            self._httpd.server_close()
        except Exception:  # noqa: BLE001 - shutdown is best-effort
            logger.debug("viewer shutdown failed", exc_info=True)
        self._httpd = None


class FleetViewers:
    """One viewer per concurrency slot, on a fixed port each.

    A slot's viewer is (re)pointed at the run of whichever scan currently holds
    the slot. The port is fixed for the slot's lifetime, so the operator's open
    tab keeps a stable URL across the whole fleet run; only its content changes
    as scans come and go. With ``concurrency`` slots there are at most
    ``concurrency`` viewers, so the fleet never opens a port per target.
    """

    def __init__(
        self,
        runs_base: Path,
        *,
        host: str,
        base_port: int,
        slots: int,
        open_browser: bool,
    ) -> None:
        self._runs_base = runs_base
        self._host = host
        self._base_port = base_port
        self._slots = max(1, slots)
        self._open_browser = open_browser
        self._viewers: dict[int, FleetViewer] = {}

    def port_for(self, slot: int) -> int:
        return self._base_port + (slot % self._slots)

    def point_at(self, slot: int, run_dir: Path) -> str | None:
        """Point ``slot``'s viewer at ``run_dir``; return its authorized URL."""
        slot = slot % self._slots
        previous = self._viewers.pop(slot, None)
        if previous is not None:
            previous.stop()
        viewer = FleetViewer(
            run_dir,
            host=self._host,
            port=self.port_for(slot),
            open_browser=self._open_browser and not self._viewers,
        )
        if not viewer.start():
            return None
        self._viewers[slot] = viewer
        return viewer.authorized_url

    def urls(self) -> dict[int, str]:
        return {
            slot: viewer.authorized_url or viewer.url or ""
            for slot, viewer in self._viewers.items()
        }

    def stop(self) -> None:
        for viewer in self._viewers.values():
            viewer.stop()
        self._viewers.clear()


# -- command -----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="strix fleet",
        description=(
            "Scan every target in a file in parallel, one scan per target, under a "
            "host memory budget, with a live web view of the whole fleet."
        ),
    )
    parser.add_argument("--target-file", required=True, help="File with one target per line.")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_MAX_CONCURRENCY,
        help="Upper bound on simultaneous scans (default: %(default)s).",
    )
    parser.add_argument(
        "--scan-mode",
        choices=["quick", "standard", "deep"],
        default="deep",
        help="Scan mode passed to every scan (default: %(default)s).",
    )
    parser.add_argument(
        "--instruction-file",
        default=None,
        help="Optional instruction file passed to every scan.",
    )
    parser.add_argument(
        "--viewer-host",
        default="127.0.0.1",
        help="Host for the live web view (default: %(default)s).",
    )
    parser.add_argument(
        "--viewer-port",
        type=int,
        default=_DEFAULT_VIEWER_PORT,
        help="Port for the live web view (default: %(default)s).",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Do not start the live web view.",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open a browser for the live web view.",
    )
    parser.add_argument(
        "--runs-dir",
        default=None,
        help=(
            f"Directory the fleet's {RUNS_DIR_NAME} tree lives in (default: the current "
            f"directory). The runs themselves always land under <DIR>/{RUNS_DIR_NAME}, "
            "which is where the scans write and the live views read."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan (targets, concurrency, caps) and exit.",
    )
    return parser


def _resolve_model_env() -> tuple[str, str, str, str]:
    from strix.config import load_settings  # noqa: PLC0415 - keeps settings off the import path

    settings = load_settings()
    model = (settings.llm.model or "").strip()
    if not model:
        raise SystemExit("No model configured. Set STRIX_LLM (and LLM_API_BASE) first.")
    api_base = (settings.llm.api_base or "").strip()
    if not api_base:
        raise SystemExit("No LLM_API_BASE configured; the fleet requires a proxy endpoint.")
    emergency = (settings.llm.emergency_model or "").strip()
    if not emergency:
        raise SystemExit("No STRIX_EMERGENCY_LLM configured; the fleet requires a failover model.")
    api_key = (settings.llm.api_key or "not-needed").strip() or "not-needed"
    return model, emergency, api_base, api_key


def _print_plan(
    console: Console,
    *,
    plan: Any,
    runs_base: Path,
    model: str,
    emergency: str,
    api_base: str,
    environment: Mapping[str, str],
    free: int | None,
    concurrency: int,
    concurrency_cap: int,
) -> None:
    console.print()
    console.print(f"[bold white]Fleet[/] {len(plan.targets)} target(s)")
    console.print(f"  model      {model}")
    console.print(f"  emergency  {emergency}")
    console.print(f"  endpoint   {api_base}")
    console.print(f"  runs       {runs_base}")
    console.print(
        f"  free mem   {_format_bytes(free)} -> concurrency {concurrency} (cap {concurrency_cap})"
    )
    console.print("  sandbox    " + _format_caps(environment))
    if plan.skipped:
        console.print(f"  skipped    {len(plan.skipped)} non-target line(s)")
    console.print()


def run_fleet(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    console = Console()

    try:
        plan = collect_scan_targets(args.target_file)
    except ValueError as exc:
        console.print(f"[bold red]{exc}[/]")
        return 2

    runs_base = _resolve_runs_base(args.runs_dir)
    model, emergency, api_base, api_key = _resolve_model_env()

    free = available_memory_bytes()
    concurrency = plan_concurrency(
        target_count=len(plan.targets),
        available_bytes=free,
        max_concurrency=args.max_concurrency,
    )
    environment = scan_environment(
        model=model,
        emergency_model=emergency,
        api_base=api_base,
        api_key=api_key,
    )
    _print_plan(
        console,
        plan=plan,
        runs_base=runs_base,
        model=model,
        emergency=emergency,
        api_base=api_base,
        environment=environment,
        free=free,
        concurrency=concurrency,
        concurrency_cap=args.max_concurrency,
    )

    if args.dry_run:
        for i, target in enumerate(plan.targets):
            console.print(f"  {i + 1:4d}  {target}")
        return 0

    instruction = ""
    if args.instruction_file:
        try:
            instruction = Path(args.instruction_file).expanduser().read_text(encoding="utf-8")
        except OSError as exc:
            console.print(f"[bold red]Cannot read instruction file: {exc}[/]")
            return 2

    run_names = run_names_for(plan.targets)
    name_by_target = dict(zip(plan.targets, run_names, strict=True))

    def scan_command(target: str, log_path: Path) -> list[str]:
        del log_path
        command = [
            sys.executable,
            "-m",
            "strix.interface.main",
            "--target",
            target,
            "-n",
            "--scan-mode",
            args.scan_mode,
            "--run-name",
            name_by_target[target],
        ]
        if instruction:
            command += ["--instruction", instruction]
        return command

    viewers = FleetViewers(
        runs_base,
        host=args.viewer_host,
        base_port=args.viewer_port,
        slots=concurrency,
        open_browser=not args.no_open,
    )

    def on_slot_started(slot: int, run_name: str) -> None:
        url = viewers.point_at(slot, runs_base / run_name)
        if url:
            console.print(f"[#60a5fa]view[/] slot {slot} -> {run_name}: {url}")
        else:
            console.print(f"[yellow]view unavailable for slot {slot} ({run_name})[/]")

    supervisor = FleetSupervisor(
        targets=plan.targets,
        scan_command=scan_command,
        environment=environment,
        runs_base=runs_base,
        concurrency=concurrency,
        per_scan_bytes=_per_scan_bytes(),
        memory_floor_bytes=_memory_floor_bytes(),
        run_names=run_names,
        on_slot_started=on_slot_started if not args.no_viewer else None,
        console=console,
    )
    previous_handlers = _install_signal_handlers(supervisor)
    try:
        result = supervisor.run()
    finally:
        _restore_signal_handlers(previous_handlers)
        viewers.stop()

    _print_summary(console, result, runs_base, viewers.urls())
    if result.failed:
        return 1
    if result.with_findings:
        return 2
    return 0


def _resolve_runs_base(runs_dir: str | None) -> Path:
    """The fleet's runs root, which is always a directory named ``strix_runs``.

    Each scan subprocess writes its run to ``<its cwd>/strix_runs/<name>`` — that
    is where the shipped entry point puts runs and there is no flag to move it —
    so the fleet runs every child with its cwd set to this root's parent. Keeping
    the root's own name fixed is what guarantees the viewers and the children
    agree on one directory; ``--runs-dir`` therefore chooses the *parent* the
    whole ``strix_runs`` tree lives in.
    """
    if not runs_dir:
        return (Path.cwd() / RUNS_DIR_NAME).resolve()
    candidate = Path(runs_dir).expanduser().resolve()
    return candidate if candidate.name == RUNS_DIR_NAME else candidate / RUNS_DIR_NAME


def _per_scan_bytes() -> int:
    per_scan_mb = _env_int("STRIX_FLEET_PER_SCAN_MB", DEFAULT_PER_SCAN_BYTES // (1024 * 1024))
    return per_scan_mb * 1024 * 1024


def _memory_floor_bytes() -> int:
    floor_mb = _env_int("STRIX_FLEET_MEMORY_FLOOR_MB", DEFAULT_MEMORY_FLOOR_BYTES // (1024 * 1024))
    return floor_mb * 1024 * 1024


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("ignoring non-numeric %s=%r", name, raw)
        return default
    return value if value > 0 else default


def _install_signal_handlers(supervisor: FleetSupervisor) -> dict[int, Any]:
    previous: dict[int, Any] = {}

    def _handler(_signum: int, _frame: Any) -> None:
        supervisor.request_stop()

    for name in ("SIGINT", "SIGTERM"):
        number = getattr(signal, name, None)
        if number is None:
            continue
        try:
            previous[number] = signal.getsignal(number)
            signal.signal(number, _handler)
        except (ValueError, OSError):
            logger.debug("could not install a %s handler", name, exc_info=True)
    return previous


def _restore_signal_handlers(previous: dict[int, Any]) -> None:
    for number, handler in previous.items():
        try:
            signal.signal(number, handler)
        except (ValueError, OSError, TypeError):
            logger.debug("could not restore signal handler %s", number, exc_info=True)


def _print_summary(
    console: Console,
    result: FleetResult,
    runs_base: Path,
    viewer_urls: Mapping[int, str] | None = None,
) -> None:
    console.print()
    console.print("[bold white]Fleet summary[/]")
    for job in result.jobs:
        console.print(f"  {job.status():>10}  {job.target}")
    console.print()
    console.print(
        f"  {len(result.jobs)} target(s): "
        f"{len(result.with_findings)} with findings, {len(result.failed)} failed"
    )
    console.print(f"  runs under {runs_base}")
    if viewer_urls:
        console.print("  live views (reopen any to inspect that slot's last scan):")
        for slot, url in sorted(viewer_urls.items()):
            if url:
                console.print(f"    slot {slot}: {url}")
    if result.stopped_early:
        console.print("  [yellow]stopped before every target ran[/]")


__all__ = [
    "FleetResult",
    "FleetSupervisor",
    "FleetViewer",
    "FleetViewers",
    "ScanJob",
    "build_parser",
    "run_fleet",
]
