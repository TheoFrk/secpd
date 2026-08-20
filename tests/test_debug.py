"""Debug-Einstellungen: ENV-Flags, Mock-Sperre, Logging, Cache-Miss-Zählung."""
from __future__ import annotations

import logging

import pytest

from secpd.config import load_settings
from secpd.llm import get_llm_client
from secpd.llm.cache import CachedLLMClient
from secpd.llm.mock import MockLLMClient


def test_debug_defaults_are_quiet_and_allow_mock(monkeypatch):
    for key in (
        "SECPD_DEBUG",
        "SECPD_LOG_LEVEL",
        "SECPD_DEBUG_KEEP_SCREEN",
        "SECPD_LLM_ALLOW_MOCK",
        "SECPD_LLM_CACHE_ONLY",
        "SECPD_LLM_FAIL_ON_MISS",
        "SECPD_LLM_MODE",
    ):
        monkeypatch.delenv(key, raising=False)
    s = load_settings()
    assert s.log_level == "OFF"
    assert s.debug_keep_screen is False
    assert s.llm_allow_mock is True
    assert s.llm_cache_only is True
    assert s.llm_fail_on_miss is False
    assert s.llm_mode == "mock"


def test_debug_flag_enables_info_and_keeps_screen(monkeypatch):
    monkeypatch.setenv("SECPD_DEBUG", "1")
    monkeypatch.delenv("SECPD_LOG_LEVEL", raising=False)
    monkeypatch.delenv("SECPD_DEBUG_KEEP_SCREEN", raising=False)
    s = load_settings()
    assert s.log_level == "INFO"
    assert s.debug_keep_screen is True


def test_keep_screen_can_be_forced_off(monkeypatch):
    monkeypatch.setenv("SECPD_DEBUG", "1")
    monkeypatch.setenv("SECPD_DEBUG_KEEP_SCREEN", "0")
    assert load_settings().debug_keep_screen is False


def test_mock_forbidden_rejects_get_llm_client(monkeypatch):
    monkeypatch.setenv("SECPD_LLM_ALLOW_MOCK", "0")
    monkeypatch.setenv("SECPD_LLM_MODE", "mock")
    with pytest.raises(ValueError, match="Mock"):
        get_llm_client()


def test_mock_forbidden_allows_openai_cache_only(monkeypatch, tmp_path):
    monkeypatch.setenv("SECPD_LLM_ALLOW_MOCK", "0")
    monkeypatch.setenv("SECPD_LLM_MODE", "openai")
    monkeypatch.setenv("SECPD_LLM_MODEL", "gpt-5.6-luna")
    monkeypatch.delenv("SECPD_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("SECPD_LLM_CACHE", str(tmp_path))
    client = get_llm_client(cache_only=True)
    assert "bank-gpt-5.6-luna" in client.name
    assert "mock" not in client.name.lower()


def test_cache_only_miss_increments_and_returns_fallback(tmp_path):
    inner = MockLLMClient()
    client = CachedLLMClient(inner, tmp_path, cache_only=True)
    profile = client.analyze("ein langer genug text für die analyse hier")
    assert client._cache_misses == 1
    assert client._cache_hits == 0
    assert profile.confidence == 0.0
    assert "cache miss" in profile.risk_summary


def test_live_miss_is_counted(tmp_path):
    inner = MockLLMClient()
    client = CachedLLMClient(inner, tmp_path, cache_only=False)
    text = (
        "Revenue for the year was 500 million dollars. Costs were 300 million. "
        "The company operates three factories and headcount was 2000 employees."
    )
    profile = client.analyze(text)
    assert client._cache_misses == 1
    assert profile.confidence > 0.0
    again = client.analyze(text)
    assert client._cache_hits == 1
    assert again == profile


def test_configure_logging_sets_info(monkeypatch):
    from secpd.cli.debug import configure_logging, reset_logging_for_tests

    monkeypatch.setenv("SECPD_LOG_LEVEL", "INFO")
    configure_logging()
    try:
        root = logging.getLogger()
        assert root.level == logging.INFO
        assert any(getattr(h, "_secpd_debug", False) for h in root.handlers)
    finally:
        reset_logging_for_tests()


def test_fail_on_miss_flag(monkeypatch):
    monkeypatch.setenv("SECPD_LLM_FAIL_ON_MISS", "ja")
    assert load_settings().llm_fail_on_miss is True


def test_raise_if_cache_miss_forbidden(monkeypatch, tmp_path):
    from secpd.cli.debug import raise_if_cache_miss_forbidden

    monkeypatch.setenv("SECPD_LLM_FAIL_ON_MISS", "1")
    client = CachedLLMClient(MockLLMClient(), tmp_path, cache_only=True)
    client.analyze("ein langer genug text für die analyse hier")
    with pytest.raises(RuntimeError, match="Cache-Miss"):
        raise_if_cache_miss_forbidden(client)
