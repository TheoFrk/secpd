"""LLM-Schicht: Schema, Client-Interface, Mock/Bank-Implementierungen, Cache.

Der einzige Unterschied zwischen Home- und Bank-Setup ist die
Umgebungsvariable ``SECPD_LLM_MODE`` (``mock`` | ``bank``) — der restliche
Code ist identisch.
"""
from __future__ import annotations

from ..config import load_settings
from .bank import BankLLMClient
from .base import BaseLLMClient, LLMResponseError
from .cache import CachedLLMClient
from .mock import MockLLMClient
from .schema import TextRiskProfile

__all__ = [
    "BaseLLMClient",
    "BankLLMClient",
    "CachedLLMClient",
    "MockLLMClient",
    "TextRiskProfile",
    "LLMResponseError",
    "get_llm_client",
]


def get_llm_client(mode: str | None = None, *, cached: bool = True) -> BaseLLMClient:
    """Factory: liefert den passenden Client für die aktuelle Umgebung.

    Parameters
    ----------
    mode:
        ``"mock"`` oder ``"bank"``. ``None`` ⇒ ``SECPD_LLM_MODE`` (Default: mock).
    cached:
        Wrappt den Client in einen Datei-Cache (empfohlen, s. ``llm.cache``).
    """
    settings = load_settings()
    resolved = (mode or settings.llm_mode).strip().lower()

    client: BaseLLMClient
    if resolved == "mock":
        client = MockLLMClient()
    elif resolved == "bank":
        client = BankLLMClient(
            endpoint=settings.llm_endpoint,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
        )
    else:
        raise ValueError(f"Unbekannter SECPD_LLM_MODE: {resolved!r} (erwartet: mock|bank)")

    if cached:
        client = CachedLLMClient(client, settings.llm_cache_dir)
    return client
