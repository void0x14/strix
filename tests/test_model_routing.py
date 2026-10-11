"""Tests for the model route a scan actually resolves.

The operator requires every model call to go through the opencodex proxy with the
primary slug, and the failover model to be only a different slug on the *same*
endpoint. These tests drive the shipped settings/model-resolution path and assert
exactly that, so a base URL that drifts to a direct provider host — or a missing
emergency model — fails here rather than in a live scan.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from strix.config import loader
from strix.config.models import StrixProvider, configure_sdk_model_defaults
from strix.core.model_failover import FailoverController, FailoverModelProvider


PROXY_BASE = "http://127.0.0.1:10100/v1"
PRIMARY = "openai/workbuddy/global:deepseek-v4.1-flash"
EMERGENCY = "openai/minimax-cn/MiniMax-M3.1-Flash-Preview"

# Hosts that would mean a scan bypassed the proxy for a direct provider.
_DIRECT_PROVIDER_HOSTS = ("workbuddy2api", "20.207.198.170", "52.172.224.33")


@pytest.fixture
def _proxy_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    # Point the config at a file that does not exist so a developer's real
    # ~/.strix/cli-config.json cannot supply a value the env does not set.
    monkeypatch.setattr(loader, "_override", tmp_path / "absent-cli-config.json")
    monkeypatch.setenv("STRIX_LLM", PRIMARY)
    monkeypatch.setenv("STRIX_EMERGENCY_LLM", EMERGENCY)
    monkeypatch.setenv("LLM_API_BASE", PROXY_BASE)
    monkeypatch.setenv("LLM_API_KEY", "sk-local-opencodex")
    monkeypatch.setattr(loader, "_cached", None)


def test_settings_resolve_the_proxy_endpoint_and_both_slugs(_proxy_env: None) -> None:
    settings = loader.load_settings()
    assert settings.llm.model == PRIMARY
    assert settings.llm.emergency_model == EMERGENCY
    assert settings.llm.api_base == PROXY_BASE
    assert not any(host in (settings.llm.api_base or "") for host in _DIRECT_PROVIDER_HOSTS)


def test_emergency_model_is_optional(_proxy_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """With no emergency model the run is unchanged: failover is opt-in."""
    monkeypatch.delenv("STRIX_EMERGENCY_LLM", raising=False)
    monkeypatch.setattr(loader, "_cached", None)
    settings = loader.load_settings()
    assert settings.llm.emergency_model is None
    assert settings.llm.model == PRIMARY
    assert settings.llm.api_base == PROXY_BASE


def test_resolved_model_client_uses_the_proxy_base_url(_proxy_env: None) -> None:
    """The model object the run config resolves talks to the proxy, not a provider."""
    settings = loader.load_settings()
    configure_sdk_model_defaults(settings)
    assert os.environ["OPENAI_BASE_URL"] == PROXY_BASE

    model = StrixProvider().get_model(settings.llm.model)
    client = _innermost_client(model)
    assert client is not None
    assert str(client.base_url).rstrip("/") == PROXY_BASE
    assert bool(getattr(client, "api_key", None))

    resolved = _innermost_model_name(model)
    assert resolved == "workbuddy/global:deepseek-v4.1-flash"
    assert not any(host in str(client.base_url) for host in _DIRECT_PROVIDER_HOSTS)


@pytest.mark.asyncio
async def test_failover_provider_calls_the_emergency_slug_on_the_same_delegate(
    _proxy_env: None,
) -> None:
    """The emergency model is only a different slug on the same endpoint."""
    settings = loader.load_settings()
    configure_sdk_model_defaults(settings)
    delegate = StrixProvider()
    requested: list[str] = []

    class _RecordingDelegate:
        def get_model(self, model_name: str | None) -> Any:
            requested.append(str(model_name))
            return delegate.get_model(model_name)

    controller = FailoverController(primary=settings.llm.model, emergency=EMERGENCY)
    provider = FailoverModelProvider(_RecordingDelegate(), controller)
    model = provider.get_model(settings.llm.model)

    # Driving a call resolves whichever slug is active through the one delegate,
    # so both slugs share its endpoint (credentials + base URL).
    await _resolve_active_slug(model, controller)
    controller.state.route = "emergency"
    await _resolve_active_slug(model, controller)

    assert requested == [PRIMARY, EMERGENCY]


async def _resolve_active_slug(model: Any, controller: Any) -> None:
    """Resolve the controller's active slug without making a network call."""
    for attempt in (model, getattr(model, "_inner", None)):
        if attempt is None:
            continue
        resolver = getattr(attempt, "_model_for", None)
        if resolver is not None:
            resolver(controller.active_model())
            return
    raise AssertionError("could not find the failover model's resolver")


def _innermost_model_name(model: Any) -> str | None:
    inner = model
    for _ in range(6):
        nxt = getattr(inner, "_inner", None) or getattr(inner, "_model", None)
        if nxt is None:
            break
        inner = nxt
    name = getattr(inner, "model", None)
    return str(name) if name is not None else None


def _innermost_client(model: Any) -> Any:
    inner = model
    for _ in range(6):
        nxt = getattr(inner, "_inner", None) or getattr(inner, "_model", None)
        if nxt is None:
            break
        inner = nxt
    return getattr(inner, "_client", None) or getattr(inner, "client", None)
