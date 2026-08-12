"""Tests für LM-Studio-/OpenAI-Endpoint-Normalisierung."""
from __future__ import annotations

from secpd.llm.bank import api_base_from_endpoint, normalize_chat_endpoint


def test_normalize_chat_endpoint_variants():
    assert normalize_chat_endpoint("172.16.3.164:1234").endswith("/v1/chat/completions")
    assert (
        normalize_chat_endpoint("http://172.16.3.164:1234")
        == "http://172.16.3.164:1234/v1/chat/completions"
    )
    assert (
        normalize_chat_endpoint("http://172.16.3.164:1234/v1")
        == "http://172.16.3.164:1234/v1/chat/completions"
    )
    full = "http://172.16.3.164:1234/v1/chat/completions"
    assert normalize_chat_endpoint(full) == full


def test_api_base_from_endpoint():
    assert api_base_from_endpoint("http://172.16.3.164:1234") == "http://172.16.3.164:1234/v1"


def test_auto_skips_embedding_models(monkeypatch):
    from secpd.llm import bank as bank_mod

    def fake_list(*_a, **_k):
        return ["text-embedding-nomic-embed-text-v1.5", "google/gemma-4-e4b"]

    monkeypatch.setattr(bank_mod, "list_openai_models", fake_list)
    assert bank_mod.resolve_model_id("http://x:1234", "auto") == "google/gemma-4-e4b"
