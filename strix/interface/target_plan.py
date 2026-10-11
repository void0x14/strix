"""Planning helpers for scanning many targets in parallel.

Scanning a list of targets is a fan-out with two hard constraints: each target
needs its own sandbox container, and the host must not be driven into OOM. This
module holds the decisions that are worth testing on their own — how the target
file becomes a target list, how many scans may run at once given the host's free
memory, and the environment a scan subprocess runs with — so the launcher script
stays a thin loop.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from strix.interface.utils import infer_target_type


if TYPE_CHECKING:
    from collections.abc import Mapping


# Per-sandbox cgroup caps. Each deep scan runs real offensive tooling inside its
# container, so the container is capped rather than trusted.
DEFAULT_SANDBOX_MEM_LIMIT = "1500m"
DEFAULT_SANDBOX_CPUS = "1.5"
DEFAULT_SANDBOX_PIDS_LIMIT = "512"
DEFAULT_SANDBOX_SHM_SIZE = "512m"
DEFAULT_SANDBOX_LOG_MAX_SIZE = "20m"

# Host memory a single scan is assumed to need: its sandbox cap plus the
# launcher-side Python/litellm heap. Used only to pick a concurrency; the
# admission check below is what actually prevents OOM.
DEFAULT_PER_SCAN_BYTES = 2 * 1024 * 1024 * 1024

# Never start a new scan unless at least this much host memory is free. A scan is
# admitted only when there is room for it, so the fleet cannot grow into an OOM;
# nothing already running is ever killed to make room.
DEFAULT_MEMORY_FLOOR_BYTES = 3 * 1024 * 1024 * 1024

DEFAULT_MAX_CONCURRENCY = 4


@dataclass(frozen=True, slots=True)
class TargetPlan:
    """The targets parsed from an input file, plus what was skipped and why."""

    targets: list[str] = field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    skipped: list[tuple[int, str]] = field(  # pyright: ignore[reportUnknownVariableType]
        default_factory=list
    )

    @property
    def total_lines(self) -> int:
        return len(self.targets) + len(self.skipped)


def _candidate_from_line(line: str) -> str:
    """The target text on a line, with a leading list marker removed.

    Target files are hand-written, so a real target may be bulleted (``- x``,
    ``* x``, ``• x``). Only the marker is stripped; the rest of the line is left
    untouched for :func:`infer_target_type` to judge.
    """
    stripped = line.strip()
    for marker in ("-", "*", "•"):
        if stripped.startswith(marker) and len(stripped) > 1 and stripped[1].isspace():
            return stripped[1:].strip()
    return stripped


def collect_scan_targets(path: str | Path) -> TargetPlan:
    """Parse a target file into the targets strix can actually scan.

    A target list is usually hand-written, so it mixes real targets with prose,
    headings, and notes. Every non-empty, non-comment line is offered to strix's
    own :func:`infer_target_type`; the lines it accepts become targets and the
    rest are recorded as skipped with their line number, so nothing is dropped
    silently. Duplicates keep their first position.

    A line containing whitespace is never a target — no URL, domain, IP, or
    repository path contains one — so it is skipped before inference. Without
    that guard a prose sentence that happens to contain a dot or a slash is
    accepted as a web target, which would send a scan at a non-asset.
    """
    source = Path(path).expanduser()
    if not source.is_file():
        raise ValueError(f"Target file '{path}' is not an existing file.")
    try:
        raw_lines = source.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"Target file '{path}' must be valid UTF-8 text: {exc}") from exc

    targets: list[str] = []
    seen: set[str] = set()
    skipped: list[tuple[int, str]] = []
    for number, line in enumerate(raw_lines, start=1):
        candidate = _candidate_from_line(line)
        if not candidate or candidate.startswith("#"):
            continue
        if any(char.isspace() for char in candidate):
            skipped.append((number, candidate))
            continue
        try:
            infer_target_type(candidate)
        except ValueError:
            skipped.append((number, candidate))
            continue
        if candidate not in seen:
            seen.add(candidate)
            targets.append(candidate)
    if not targets:
        raise ValueError(f"Target file '{path}' yielded no scannable targets.")
    return TargetPlan(targets=targets, skipped=skipped)


def available_memory_bytes(meminfo_path: str | Path = "/proc/meminfo") -> int | None:
    """Host memory available for new work, or None when it cannot be read."""
    try:
        text = Path(meminfo_path).read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1]) * 1024
    return None


def plan_concurrency(
    *,
    target_count: int,
    available_bytes: int | None,
    per_scan_bytes: int = DEFAULT_PER_SCAN_BYTES,
    memory_floor_bytes: int = DEFAULT_MEMORY_FLOOR_BYTES,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> int:
    """How many scans may run at once without risking an OOM.

    The headroom is what is free *after* leaving the floor untouched, divided by
    what one scan needs. The result is at least one (a single scan is always
    allowed — it is the operator's explicit request) and never more than
    ``max_concurrency`` or the number of targets.
    """
    if target_count <= 0:
        return 0
    if available_bytes is None:
        return min(max_concurrency, target_count, 1)
    headroom = available_bytes - memory_floor_bytes
    if headroom <= 0:
        return 1
    return max(1, min(max_concurrency, target_count, headroom // per_scan_bytes))


def has_memory_headroom(
    *,
    available_bytes: int | None,
    per_scan_bytes: int = DEFAULT_PER_SCAN_BYTES,
    memory_floor_bytes: int = DEFAULT_MEMORY_FLOOR_BYTES,
) -> bool:
    """Whether one more scan can start right now.

    When the host's free memory cannot be read the check is permissive: an
    unreadable ``/proc/meminfo`` is not evidence of pressure, and refusing to
    start anything would be worse than the risk it guards against.
    """
    if available_bytes is None:
        return True
    return available_bytes - per_scan_bytes >= memory_floor_bytes


def sandbox_resource_env(
    *,
    mem_limit: str = DEFAULT_SANDBOX_MEM_LIMIT,
    cpus: str = DEFAULT_SANDBOX_CPUS,
    pids_limit: str = DEFAULT_SANDBOX_PIDS_LIMIT,
    shm_size: str = DEFAULT_SANDBOX_SHM_SIZE,
    log_max_size: str = DEFAULT_SANDBOX_LOG_MAX_SIZE,
) -> dict[str, str]:
    """The cgroup caps every launched sandbox must carry.

    These are read by the Docker backend when it creates the container, so they
    are part of the scan subprocess's environment.
    """
    return {
        "STRIX_SANDBOX_MEM_LIMIT": mem_limit,
        "STRIX_SANDBOX_CPUS": cpus,
        "STRIX_SANDBOX_PIDS_LIMIT": pids_limit,
        "STRIX_SANDBOX_SHM_SIZE": shm_size,
        "STRIX_SANDBOX_LOG_MAX_SIZE": log_max_size,
    }


def scan_environment(
    *,
    model: str,
    emergency_model: str,
    api_base: str,
    api_key: str,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """The environment a scan subprocess runs with.

    The model env is the only source of the model route, so it is set here
    explicitly: the primary slug and the emergency slug both resolve through
    ``api_base``, which is the proxy endpoint — never a direct provider host.
    """
    env = dict(os.environ if base is None else base)
    env["STRIX_LLM"] = model
    env["STRIX_EMERGENCY_LLM"] = emergency_model
    env["LLM_API_BASE"] = api_base
    env["LLM_API_KEY"] = api_key
    # The API base is OpenAI-compatible, so the chat-completions tool schema is
    # what the endpoint accepts.
    env["STRIX_API_TYPE"] = "chat_completions"
    env.update(sandbox_resource_env())
    return env


def slugify_target(target: str, *, max_length: int = 48) -> str:
    """A filesystem-safe label for a target, used for logs and report files."""
    text = target.lower().strip()
    text = re.sub(r"^[a-z][a-z0-9+.-]*://", "", text)
    text = re.sub(r"[^a-z0-9.]+", "-", text).strip("-.")
    return (text[:max_length].rstrip("-.")) or "target"


def run_names_for(targets: list[str]) -> list[str]:
    """One stable run name per target, unique even when two targets share a slug.

    A fleet gives every target its own run directory, so the name has to be
    stable (a rerun lands in the same directory and resumes) and unique (two
    targets that slugify alike must not collide).
    """
    slugs = [slugify_target(target) for target in targets]
    counts: dict[str, int] = {}
    for slug in slugs:
        counts[slug] = counts.get(slug, 0) + 1
    used: set[str] = set()
    names: list[str] = []
    for index, slug in enumerate(slugs):
        name = f"fleet-{slug}"
        if counts[slug] > 1 or name in used:
            name = f"fleet-{slug}-{index}"
        used.add(name)
        names.append(name)
    return names


__all__ = [
    "DEFAULT_MAX_CONCURRENCY",
    "DEFAULT_MEMORY_FLOOR_BYTES",
    "DEFAULT_PER_SCAN_BYTES",
    "DEFAULT_SANDBOX_CPUS",
    "DEFAULT_SANDBOX_LOG_MAX_SIZE",
    "DEFAULT_SANDBOX_MEM_LIMIT",
    "DEFAULT_SANDBOX_PIDS_LIMIT",
    "DEFAULT_SANDBOX_SHM_SIZE",
    "TargetPlan",
    "available_memory_bytes",
    "collect_scan_targets",
    "has_memory_headroom",
    "plan_concurrency",
    "run_names_for",
    "sandbox_resource_env",
    "scan_environment",
    "slugify_target",
]
