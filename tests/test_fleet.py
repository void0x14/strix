"""Tests for the parallel fleet planner and supervisor.

The planner decisions (target parsing, concurrency, resource caps, environment)
are pure and asserted directly. The supervisor's admission loop is driven with
injected memory readings, a fake launcher, and a fake clock, so the OOM guard
and the bounded concurrency are exercised without Docker or a real scan.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from strix.interface import fleet as fleet_module
from strix.interface.target_plan import (
    DEFAULT_MEMORY_FLOOR_BYTES,
    DEFAULT_PER_SCAN_BYTES,
    DEFAULT_SANDBOX_CPUS,
    DEFAULT_SANDBOX_MEM_LIMIT,
    DEFAULT_SANDBOX_PIDS_LIMIT,
    TargetPlan,
    available_memory_bytes,
    collect_scan_targets,
    has_memory_headroom,
    plan_concurrency,
    run_names_for,
    sandbox_resource_env,
    scan_environment,
    slugify_target,
)


GIB = 1024**3


# --- target parsing ---------------------------------------------------------


def test_collect_scan_targets_keeps_every_target_and_reports_the_rest(tmp_path: Path) -> None:
    target_file = tmp_path / "targets.md"
    lines = [
        "# heading",
        "",
        "- URL",
        "--Domain--",
        "example.com",
        "https://app.example.com/login",
        "192.168.1.10",
        "- example.org",
        "This is a note about the asset, not a target.",
        "https://github.com/org/repo",
        "example.com",  # duplicate: keeps its first position
        "",
    ]
    target_file.write_text("\n".join(lines), encoding="utf-8")
    plan = collect_scan_targets(target_file)
    assert plan.targets == [
        "example.com",
        "https://app.example.com/login",
        "192.168.1.10",
        "example.org",
        "https://github.com/org/repo",
    ]
    skipped_lines = {number for number, _ in plan.skipped}
    assert 3 in skipped_lines  # "- URL"
    assert 4 in skipped_lines  # "--Domain--"
    assert 9 in skipped_lines  # the prose note
    assert plan.total_lines == 5 + len(plan.skipped)


def test_collect_scan_targets_skips_prose_that_looks_like_a_host(tmp_path: Path) -> None:
    """A sentence containing a dot must not become a scan target."""
    target_file = tmp_path / "targets.txt"
    target_file.write_text(
        "Our main app judge.me is the core of the system.\njudge.me\n",
        encoding="utf-8",
    )
    plan = collect_scan_targets(target_file)
    assert plan.targets == ["judge.me"]


def test_collect_scan_targets_rejects_an_empty_result(tmp_path: Path) -> None:
    target_file = tmp_path / "targets.txt"
    target_file.write_text("just a note\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no scannable targets"):
        collect_scan_targets(target_file)


def test_collect_scan_targets_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not an existing file"):
        collect_scan_targets(tmp_path / "nope.txt")


def test_real_target_file_yields_every_asset() -> None:
    """The operator's real target file parses to its full target set."""
    source = Path("/home/void0x14/Documents/111026-HEDEFLER.md")
    if not source.is_file():
        pytest.skip("operator target file not present on this host")
    plan = collect_scan_targets(source)
    assert len(plan.targets) >= 100
    # Nothing that carries whitespace ever becomes a target.
    assert not any(" " in target for target in plan.targets)
    # A known URL and a known bare domain are both present.
    assert "https://github.com/circlefin/aptos-cctp" in plan.targets
    assert "judge.me" in plan.targets


# --- concurrency / OOM guard ------------------------------------------------


def test_plan_concurrency_scales_with_free_memory() -> None:
    assert plan_concurrency(target_count=100, available_bytes=11 * GIB, max_concurrency=4) == 4
    # ~5 GiB free leaves ~2 GiB of headroom above the floor: one scan.
    assert plan_concurrency(target_count=100, available_bytes=5 * GIB, max_concurrency=4) == 1


def test_plan_concurrency_is_bounded_by_target_count_and_cap() -> None:
    assert plan_concurrency(target_count=1, available_bytes=64 * GIB, max_concurrency=8) == 1
    assert plan_concurrency(target_count=100, available_bytes=64 * GIB, max_concurrency=3) == 3
    assert plan_concurrency(target_count=0, available_bytes=64 * GIB, max_concurrency=3) == 0


def test_plan_concurrency_never_starves_a_scan() -> None:
    """Under pressure one scan still runs: the operator asked for the scans."""
    assert plan_concurrency(target_count=50, available_bytes=1 * GIB, max_concurrency=4) == 1


def test_has_memory_headroom_respects_the_floor() -> None:
    assert has_memory_headroom(available_bytes=11 * GIB) is True
    assert has_memory_headroom(available_bytes=DEFAULT_MEMORY_FLOOR_BYTES) is False
    assert has_memory_headroom(available_bytes=DEFAULT_MEMORY_FLOOR_BYTES + DEFAULT_PER_SCAN_BYTES)
    # An unreadable reading is not evidence of pressure.
    assert has_memory_headroom(available_bytes=None) is True


def test_available_memory_bytes_reads_meminfo(tmp_path: Path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:       16384000 kB\nMemAvailable:    8192000 kB\n", encoding="utf-8"
    )
    assert available_memory_bytes(meminfo) == 8192000 * 1024


def test_available_memory_bytes_is_none_when_unreadable(tmp_path: Path) -> None:
    assert available_memory_bytes(tmp_path / "missing") is None


# --- resource caps ----------------------------------------------------------


def test_sandbox_resource_env_caps_every_sandbox() -> None:
    env = sandbox_resource_env()
    assert env["STRIX_SANDBOX_MEM_LIMIT"] == DEFAULT_SANDBOX_MEM_LIMIT
    assert env["STRIX_SANDBOX_CPUS"] == DEFAULT_SANDBOX_CPUS
    assert env["STRIX_SANDBOX_PIDS_LIMIT"] == DEFAULT_SANDBOX_PIDS_LIMIT
    assert env["STRIX_SANDBOX_LOG_MAX_SIZE"]
    # Every cap the docker backend reads is present.
    assert set(env) == {
        "STRIX_SANDBOX_MEM_LIMIT",
        "STRIX_SANDBOX_CPUS",
        "STRIX_SANDBOX_PIDS_LIMIT",
        "STRIX_SANDBOX_SHM_SIZE",
        "STRIX_SANDBOX_LOG_MAX_SIZE",
    }


def test_docker_backend_applies_the_caps_to_create_kwargs() -> None:
    """The caps reach the container, not just the environment."""
    from strix.runtime.docker_client import (  # noqa: PLC0415 - docker SDK is heavy
        _apply_resource_limits,
    )

    caps = sandbox_resource_env()
    create_kwargs: dict[str, Any] = {}
    saved = {key: os.environ.get(key) for key in caps}
    os.environ.update(caps)
    try:
        _apply_resource_limits(create_kwargs)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    assert create_kwargs["mem_limit"] == DEFAULT_SANDBOX_MEM_LIMIT
    assert create_kwargs["pids_limit"] == int(DEFAULT_SANDBOX_PIDS_LIMIT)
    assert create_kwargs["nano_cpus"] == int(float(DEFAULT_SANDBOX_CPUS) * 1_000_000_000)


# --- environment ------------------------------------------------------------


def test_scan_environment_routes_both_models_through_the_proxy() -> None:
    env = scan_environment(
        model="openai/workbuddy/global:deepseek-v4.1-flash",
        emergency_model="openai/minimax-cn/MiniMax-M3.1-Flash-Preview",
        api_base="http://127.0.0.1:10100/v1",
        api_key="sk-local-opencodex",
        base={},
    )
    assert env["STRIX_LLM"] == "openai/workbuddy/global:deepseek-v4.1-flash"
    assert env["STRIX_EMERGENCY_LLM"] == "openai/minimax-cn/MiniMax-M3.1-Flash-Preview"
    assert env["LLM_API_BASE"] == "http://127.0.0.1:10100/v1"
    assert env["STRIX_API_TYPE"] == "chat_completions"
    # No direct provider host anywhere.
    assert "workbuddy2api" not in str(env)
    # The sandbox caps travel with the same environment.
    assert env["STRIX_SANDBOX_MEM_LIMIT"] == DEFAULT_SANDBOX_MEM_LIMIT


# --- run names / slugs ------------------------------------------------------


def test_run_names_are_stable_and_unique() -> None:
    targets = ["https://a.example.com/x", "https://b.example.com/x", "https://a.example.com/x"]
    names = run_names_for(targets)
    assert len(set(names)) == 3
    assert names[0].startswith("fleet-")
    # Stable: the same input yields the same names.
    assert run_names_for(targets) == names


def test_slugify_target_strips_the_scheme() -> None:
    assert slugify_target("https://app.example.com/login?x=1") == "app.example.com-login-x-1"
    assert slugify_target("*.hypr.com") == "hypr.com"
    assert slugify_target("") == "target"


# --- supervisor admission loop ----------------------------------------------


class _FakeProcess:
    def __init__(self, *, exit_after: int) -> None:
        self.polls = 0
        self.exit_after = exit_after
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        self.polls += 1
        return 0 if self.polls > self.exit_after else None

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 0.5
        return self.now


def _supervisor(
    tmp_path: Path,
    *,
    targets: list[str],
    concurrency: int,
    memory_reader: Any,
    launcher: Any,
    per_scan_bytes: int = DEFAULT_PER_SCAN_BYTES,
) -> fleet_module.FleetSupervisor:
    return fleet_module.FleetSupervisor(
        targets=targets,
        scan_command=lambda target, _log_path: ["echo", target],
        environment={"A": "1"},
        runs_base=tmp_path / "strix_runs",
        concurrency=concurrency,
        per_scan_bytes=per_scan_bytes,
        memory_floor_bytes=DEFAULT_MEMORY_FLOOR_BYTES,
        memory_reader=memory_reader,
        launcher=launcher,
        clock=_FakeClock(),
        sleep=lambda _seconds: None,
        console=_QuietConsole(),
    )


class _QuietConsole:
    def print(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class _RecordingConsole:
    def __init__(self, messages: list[str]) -> None:
        self._messages = messages

    def print(self, *args: Any, **_kwargs: Any) -> None:
        self._messages.extend(str(arg) for arg in args)


def test_supervisor_never_exceeds_its_concurrency(tmp_path: Path) -> None:
    live: list[_FakeProcess] = []

    def _launcher(*_args: Any, **_kwargs: Any) -> _FakeProcess:
        process = _FakeProcess(exit_after=2)
        live.append(process)
        return process

    supervisor = _supervisor(
        tmp_path,
        targets=[f"t{i}.example.com" for i in range(6)],
        concurrency=2,
        memory_reader=lambda: 64 * GIB,
        launcher=_launcher,
    )
    # At every point the supervisor can be observed, no more than 2 are live.
    observed = 0

    def _observe() -> None:
        nonlocal observed
        observed = max(observed, supervisor._active_count())

    original_start = supervisor._start

    def _wrapped_start(job: Any) -> None:
        original_start(job)
        _observe()

    supervisor._start = _wrapped_start  # type: ignore[method-assign]
    result = supervisor.run()
    assert observed <= 2
    assert len(result.jobs) == 6
    assert all(job.returncode == 0 for job in result.jobs)


def test_supervisor_holds_new_scans_when_memory_is_short(tmp_path: Path) -> None:
    """Under memory pressure a new scan is never started until headroom returns."""
    started: list[str] = []
    free = {"bytes": 64 * GIB}

    def _launcher(command: list[str], **_kwargs: Any) -> _FakeProcess:
        started.append(command[-1])
        return _FakeProcess(exit_after=0)

    def _reader() -> int:
        return free["bytes"]

    supervisor = _supervisor(
        tmp_path,
        targets=[f"t{i}.example.com" for i in range(4)],
        concurrency=4,
        memory_reader=_reader,
        launcher=_launcher,
    )
    original_start = supervisor._start
    ticks = {"n": 0}

    def _wrapped_start(job: Any) -> None:
        original_start(job)
        if len(started) == 1:
            # Memory collapses right after the first scan starts, so no further
            # scan may be admitted.
            free["bytes"] = DEFAULT_MEMORY_FLOOR_BYTES

    def _sleep(_seconds: float) -> None:
        ticks["n"] += 1
        # Once the collapsed reading has been observed, memory frees up again.
        if free["bytes"] == DEFAULT_MEMORY_FLOOR_BYTES and ticks["n"] >= 2:
            free["bytes"] = 64 * GIB

    supervisor._start = _wrapped_start  # type: ignore[method-assign]
    supervisor._sleep = _sleep
    result = supervisor.run()

    # Every target eventually ran, and no scan was ever started while the host
    # was under the floor.
    assert started == ["t0.example.com", "t1.example.com", "t2.example.com", "t3.example.com"]
    assert all(job.returncode == 0 for job in result.jobs)
    assert ticks["n"] >= 2


def test_supervisor_reports_a_stalled_admission_instead_of_spinning(
    tmp_path: Path,
) -> None:
    """When nothing can be admitted and nothing runs, the fleet says so."""
    started: list[str] = []

    def _launcher(command: list[str], **_kwargs: Any) -> _FakeProcess:
        started.append(command[-1])
        return _FakeProcess(exit_after=0)

    supervisor = _supervisor(
        tmp_path,
        targets=["a.example.com"],
        concurrency=1,
        memory_reader=lambda: DEFAULT_MEMORY_FLOOR_BYTES,
        launcher=_launcher,
    )
    messages: list[str] = []
    supervisor._console = _RecordingConsole(messages)
    # Stop after the first poll so the stalled state is observed once.
    ticks = {"n": 0}

    def _sleep(_seconds: float) -> None:
        ticks["n"] += 1
        if ticks["n"] >= 1:
            supervisor.request_stop()

    supervisor._sleep = _sleep
    result = supervisor.run()
    assert started == []
    assert any("waiting for memory headroom" in message for message in messages)
    assert result.stopped_early is True


def test_supervisor_refuses_admission_at_or_below_the_floor(tmp_path: Path) -> None:
    """The admission gate itself is what holds a scan, tested without the loop."""
    started: list[str] = []

    def _launcher(command: list[str], **_kwargs: Any) -> _FakeProcess:
        started.append(command[-1])
        return _FakeProcess(exit_after=0)

    supervisor = _supervisor(
        tmp_path,
        targets=["a.example.com", "b.example.com"],
        concurrency=4,
        memory_reader=lambda: DEFAULT_MEMORY_FLOOR_BYTES,
        launcher=_launcher,
    )
    assert supervisor._admit() is False
    assert started == []

    supervisor._memory_reader = lambda: 64 * GIB
    assert supervisor._admit() is True


def test_supervisor_marks_a_launch_failure_without_stopping_the_fleet(tmp_path: Path) -> None:
    calls = {"n": 0}

    def _launcher(*_args: Any, **_kwargs: Any) -> _FakeProcess:
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("docker daemon is gone")
        return _FakeProcess(exit_after=0)

    supervisor = _supervisor(
        tmp_path,
        targets=["a.example.com", "b.example.com"],
        concurrency=1,
        memory_reader=lambda: 64 * GIB,
        launcher=_launcher,
    )
    result = supervisor.run()
    assert result.jobs[0].returncode == -1
    assert result.jobs[1].returncode == 0
    assert len(result.failed) == 1


def test_supervisor_reports_findings_and_failures(tmp_path: Path) -> None:
    codes = iter([0, 2, 1])

    def _launcher(*_args: Any, **_kwargs: Any) -> _FakeProcess:
        code = next(codes)

        class _P(_FakeProcess):
            def poll(self) -> int | None:
                return code

        return _P(exit_after=0)

    supervisor = _supervisor(
        tmp_path,
        targets=[f"t{i}.example.com" for i in range(3)],
        concurrency=1,
        memory_reader=lambda: 64 * GIB,
        launcher=_launcher,
    )
    result = supervisor.run()
    assert [job.returncode for job in result.jobs] == [0, 2, 1]
    assert len(result.with_findings) == 1
    assert len(result.failed) == 1


def test_supervisor_stop_terminates_running_scans(tmp_path: Path) -> None:
    processes: list[_FakeProcess] = []

    def _launcher(*_args: Any, **_kwargs: Any) -> _FakeProcess:
        process = _FakeProcess(exit_after=10_000)
        processes.append(process)
        return process

    supervisor = _supervisor(
        tmp_path,
        targets=["a.example.com", "b.example.com"],
        concurrency=2,
        memory_reader=lambda: 64 * GIB,
        launcher=_launcher,
    )
    original_start = supervisor._start

    def _wrapped_start(job: Any) -> None:
        original_start(job)
        supervisor.request_stop()

    supervisor._start = _wrapped_start  # type: ignore[method-assign]
    result = supervisor.run()
    assert result.stopped_early is True
    assert all(process.terminated for process in processes)


def test_plan_dataclass_totals() -> None:
    plan = TargetPlan(targets=["a.com", "b.com"], skipped=[(1, "note")])
    assert plan.total_lines == 3


# --- resume: a restarted fleet must not redo finished targets -----------------


def test_resume_skips_targets_with_a_finished_run(tmp_path: Path) -> None:
    runs_base = tmp_path / "strix_runs"
    # target-a finished; target-b never started; target-c was interrupted.
    for slug, status, end_time in (
        ("fleet-a.example.com", "completed", "2026-01-01T00:00:00Z"),
        ("fleet-c.example.com", "interrupted", "2026-01-01T00:00:00Z"),
    ):
        run_dir = runs_base / slug
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text(
            json.dumps({"run_name": slug, "status": status, "end_time": end_time}),
            encoding="utf-8",
        )

    started: list[str] = []

    def _launcher(command: list[str], **_kwargs: Any) -> _FakeProcess:
        started.append(command[-1])
        return _FakeProcess(exit_after=0)

    # Resume on: the finished targets are skipped, the rest run.
    resumed = fleet_module.FleetSupervisor(
        targets=["a.example.com", "b.example.com", "c.example.com"],
        scan_command=lambda target, _log_path: ["echo", target],
        environment={"A": "1"},
        runs_base=runs_base,
        concurrency=1,
        per_scan_bytes=DEFAULT_PER_SCAN_BYTES,
        memory_floor_bytes=DEFAULT_MEMORY_FLOOR_BYTES,
        memory_reader=lambda: 64 * GIB,
        launcher=_launcher,
        clock=_FakeClock(),
        sleep=lambda _s: None,
        console=_QuietConsole(),
        resume=True,
    )
    result = resumed.run()

    # a.example.com and c.example.com are already settled and must not rerun.
    assert started == ["b.example.com"]
    statuses = {job.target: job.status() for job in result.jobs}
    assert statuses["a.example.com"] == "skipped"
    assert statuses["c.example.com"] == "skipped"
    assert statuses["b.example.com"] == "exit 0"


def test_without_resume_a_finished_target_is_rerun(tmp_path: Path) -> None:
    """The default is unchanged: a plain launch scans every target."""
    runs_base = tmp_path / "strix_runs"
    run_dir = runs_base / "fleet-a.example.com"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps({"status": "completed", "end_time": "2026-01-01T00:00:00Z"}),
        encoding="utf-8",
    )
    started: list[str] = []

    def _launcher(command: list[str], **_kwargs: Any) -> _FakeProcess:
        started.append(command[-1])
        return _FakeProcess(exit_after=0)

    supervisor = _supervisor(
        tmp_path,
        targets=["a.example.com"],
        concurrency=1,
        memory_reader=lambda: 64 * GIB,
        launcher=_launcher,
    )
    supervisor._runs_base = runs_base
    supervisor.run()
    assert started == ["a.example.com"]
