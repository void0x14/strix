"""Tests for the fleet's per-slot viewers.

The bug this guards: a viewer's session token is minted per process, so a slot
that *restarts* its viewer to follow a new target mints a new token and every
URL the operator already has open starts answering 403. The slot viewer must
therefore be started once and only retargeted.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from http.cookiejar import CookieJar
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import pytest

from strix.interface.fleet import FleetViewer, FleetViewers
from strix.interface.viewer.server import bundle_is_built


if TYPE_CHECKING:
    from pathlib import Path


pytestmark = pytest.mark.skipif(
    not bundle_is_built(), reason="viewer bundle is not built in this checkout"
)


def _make_run(base: Path, name: str, *, status: str = "running") -> Path:
    run_dir = base / "strix_runs" / name
    state = run_dir / ".state"
    state.mkdir(parents=True)
    record = {"run_name": name, "status": status, "end_time": None, "scan_mode": "deep"}
    (run_dir / "run.json").write_text(json.dumps(record), encoding="utf-8")
    return run_dir


def _client() -> Any:
    jar = CookieJar()
    return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar)), jar


def _get(opener: Any, url: str) -> tuple[int, bytes]:
    try:
        with opener.open(url, timeout=10) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def test_slot_viewer_keeps_its_token_across_a_retarget(tmp_path: Path) -> None:
    """An open tab follows the slot's next target instead of being logged out."""
    first = _make_run(tmp_path, "first-target")
    second = _make_run(tmp_path, "second-target")

    viewer = FleetViewer(first, host="127.0.0.1", port=0, open_browser=False)
    assert viewer.start(first) is True
    url = viewer.authorized_url
    assert url is not None

    opener, jar = _client()
    status, _ = _get(opener, url)
    assert status == 200
    cookie = jar._cookies["127.0.0.1"]["/"]["strix_viewer_session_" + str(_port_of(url))]
    token = cookie.value
    status, body = _get(opener, f"{_base_of(url)}/api/run")
    assert status == 200
    assert json.loads(body)["run_name"] == "first-target"

    # The slot's scan finished; the next target now holds the slot.
    assert viewer.retarget(second) is True

    # The same URL and the same cookie still work, and now show the new run.
    assert viewer.authorized_url == url
    status, body = _get(opener, f"{_base_of(url)}/api/run")
    assert status == 200, "retargeting must not invalidate the operator's session"
    assert json.loads(body)["run_name"] == "second-target"
    # The session token itself is unchanged.
    cookie_after = jar._cookies["127.0.0.1"]["/"]["strix_viewer_session_" + str(_port_of(url))]
    assert cookie_after.value == token
    viewer.stop()


def test_slot_viewers_keep_one_port_and_one_token_per_slot(tmp_path: Path) -> None:
    """A slot re-pointed many times keeps its port and token: no 403 storm."""
    runs = [_make_run(tmp_path, f"target-{i}") for i in range(4)]
    viewers = FleetViewers(
        tmp_path / "strix_runs", host="127.0.0.1", base_port=0, slots=1, open_browser=False
    )
    # base_port 0 would make every slot ephemeral; use an explicit port via slot 0.
    first_url = viewers.point_at(0, runs[0])
    assert first_url is not None
    opener, _jar = _client()
    status, _ = _get(opener, first_url)
    assert status == 200

    for run in runs[1:]:
        url = viewers.point_at(0, run)
        # Same port, same token: the URL string never changes.
        assert url == first_url
        status, body = _get(opener, f"{_base_of(url)}/api/run")
        assert status == 200
        assert json.loads(body)["run_name"] == run.name
    viewers.stop()


def _port_of(authorized_url: str) -> int:
    return urlsplit(authorized_url).port or 0


def _base_of(authorized_url: str) -> str:
    parts = urlsplit(authorized_url)
    return f"{parts.scheme}://{parts.netloc}"
