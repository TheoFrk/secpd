"""LLM-Schicht: Mock, LM Studio, OpenAI/ChatGPT, Bank-Gateway + Datei-Cache.

Modi über ``SECPD_LLM_MODE``:
``mock`` | ``lmstudio`` | ``openai``/``chatgpt`` | ``bank``
"""
from __future__ import annotations

import logging
import os

from ..config import load_settings
from .bank import (
    DEFAULT_LMSTUDIO_HOST,
    DEFAULT_OPENAI_ENDPOINT,
    DEFAULT_OPENAI_MODEL,
    BankLLMClient,
)
from .base import BaseLLMClient, LLMResponseError
from .cache import CachedLLMClient
from .mock import MockLLMClient
from .schema import TextRiskProfile

logger = logging.getLogger(__name__)

__all__ = [
    "BaseLLMClient",
    "BankLLMClient",
    "CachedLLMClient",
    "MockLLMClient",
    "TextRiskProfile",
    "LLMResponseError",
    "get_llm_client",
    "DEFAULT_LMSTUDIO_HOST",
    "DEFAULT_OPENAI_ENDPOINT",
    "DEFAULT_OPENAI_MODEL",
]


def _env_bool(name: str, default: bool | None = None) -> bool | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def get_llm_client(
    mode: str | None = None,
    *,
    cached: bool = True,
    force_refresh: bool = False,
    cache_only: bool = False,
) -> BaseLLMClient:
    """Factory für den LLM-Client.

    Parameters
    ----------
    mode:
        ``mock`` | ``lmstudio`` | ``openai``/``chatgpt`` | ``bank``.
    cached:
        Datei-Cache unter ``data/cache/llm/<namespace>/``.
    force_refresh:
        Bei Cache: vorhandene Einträge ignorieren und neu bewerten (danach
        wieder speichern).
    cache_only:
        Keine LLM-API bei Cache-Miss — Fallback-Profil. Wird automatisch
        gesetzt, wenn OpenAI ohne API-Key und ohne ``force_refresh``.
    """
    settings = load_settings()
    resolved = (mode or settings.llm_mode).strip().lower()

    client: BaseLLMClient
    if resolved == "mock":
        client = MockLLMClient()
    elif resolved in {"openai", "chatgpt", "gpt"}:
        api_key = settings.llm_api_key or os.getenv("OPENAI_API_KEY", "")
        if force_refresh and not api_key:
            raise ValueError(
                "OpenAI-Refresh braucht SECPD_LLM_API_KEY (oder OPENAI_API_KEY) — "
                "in start.py → Einstellungen → LLM setzen."
            )
        if not api_key:
            cache_only = True
            api_key = "cache-only"
            logger.info("OpenAI ohne API-Key — Cache-only Replay.")
        endpoint = settings.llm_endpoint or DEFAULT_OPENAI_ENDPOINT
        model = settings.llm_model
        if not model or model in {"internal-default", "auto", "local"}:
            model = DEFAULT_OPENAI_MODEL
        timeout = float(os.getenv("SECPD_LLM_TIMEOUT", "120"))
        json_mode = _env_bool("SECPD_LLM_JSON_MODE", default=True)
        client = BankLLMClient(
            endpoint=endpoint,
            api_key=api_key,
            model=model,
            timeout=timeout,
            json_mode=json_mode,
            resolve_model=False,
        )
    elif resolved in {"bank", "lmstudio", "lms"}:
        endpoint = settings.llm_endpoint
        model = settings.llm_model
        api_key = settings.llm_api_key
        timeout = float(os.getenv("SECPD_LLM_TIMEOUT", "300"))
        json_mode = _env_bool("SECPD_LLM_JSON_MODE", default=None)
        if resolved in {"lmstudio", "lms"} or not endpoint:
            endpoint = endpoint or DEFAULT_LMSTUDIO_HOST
            api_key = api_key or "lm-studio"
            if not settings.llm_model or settings.llm_model == "internal-default":
                model = "auto"
            if json_mode is None:
                json_mode = False
            if resolved == "bank" and not settings.llm_endpoint:
                logger.warning(
                    "SECPD_LLM_MODE=bank ohne ENDPOINT — Fallback auf LM Studio %s",
                    endpoint,
                )
        client = BankLLMClient(
            endpoint=endpoint,
            api_key=api_key,
            model=model,
            timeout=timeout,
            json_mode=json_mode,
        )
    else:
        raise ValueError(
            f"Unbekannter SECPD_LLM_MODE: {resolved!r} "
            "(erwartet: mock|lmstudio|openai|bank)"
        )

    if cached:
        client = CachedLLMClient(
            client,
            settings.llm_cache_dir,
            force_refresh=force_refresh,
            cache_only=cache_only,
        )
    return client
