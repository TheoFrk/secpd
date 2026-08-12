"""Echter LLM-Client für OpenAI-kompatible Gateways (Bank oder LM Studio).

Annahme: POST ``/v1/chat/completions``. Konfiguration über ENV:

* ``SECPD_LLM_ENDPOINT`` — Host oder volle Completions-URL
* ``SECPD_LLM_API_KEY``  — Key (LM Studio akzeptiert oft einen Platzhalter)
* ``SECPD_LLM_MODEL``    — Modell-ID (``auto`` ⇒ erstes Modell aus ``/v1/models``)
* ``SECPD_LLM_JSON_MODE`` — ``1``/``0``; ``response_format=json_object`` (manche
  lokalen Modelle verweigern das)

Robustheit: JSON-Fences werden entfernt, Pydantic validiert strikt, bei
Validierungsfehlern erfolgt genau **ein** Reparatur-Roundtrip, danach wird
``LLMResponseError`` geworfen (der Batch-Layer fängt das per Fallback ab).
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any
from urllib.parse import urlparse, urlunparse

import requests
from pydantic import ValidationError

from .base import BaseLLMClient, LLMResponseError
from .schema import TextRiskProfile

logger = logging.getLogger(__name__)

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

DEFAULT_LMSTUDIO_HOST = "http://172.16.3.164:1234"
DEFAULT_OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"
DEFAULT_OPENAI_MODEL = "gpt-5.6-luna"

SYSTEM_PROMPT = (
    "You are a senior credit risk analyst reviewing SEC filing excerpts "
    "(10-K/8-K). Assess the disclosure quality and risk tone of the given text. "
    "Respond with a SINGLE JSON object only — no prose, no markdown fences."
)

USER_PROMPT_TEMPLATE = """Analyze the following filing excerpt.

Return ONLY a JSON object with exactly these fields:
- vagueness_score: float in [0,1] (0=precise/quantified, 1=maximally evasive)
- redundancy_score: float in [0,1] (share of repetitive/boilerplate language)
- complexity_score: float in [0,1] (sentence length, nesting, jargon density)
- risk_sentiment: float in [-1,1] (-1=crisis-like negative tone, +1=confident)
- confidence: float in [0,1] (your confidence in this assessment)
- obfuscation_indicators: list of up to 5 short keyword phrases
- risk_summary: string, at most 3 sentences

JSON schema (informative):
{schema}

Filing excerpt:
<<<
{text}
>>>"""


def _strip_fences(content: str) -> str:
    return _FENCE.sub("", content).strip()


def normalize_chat_endpoint(endpoint: str) -> str:
    """Akzeptiert Host (`http://ip:1234`) oder volle Chat-URL."""
    ep = (endpoint or "").strip().rstrip("/")
    if not ep:
        return ep
    if not ep.startswith(("http://", "https://")):
        ep = "http://" + ep
    parsed = urlparse(ep)
    path = (parsed.path or "").rstrip("/")
    if path.endswith("/chat/completions"):
        return ep
    if path.endswith("/v1"):
        return ep + "/chat/completions"
    if path in ("", "/"):
        return urlunparse((parsed.scheme, parsed.netloc, "/v1/chat/completions", "", "", ""))
    # Unbekannter Pfad: unverändert lassen
    return ep


def api_base_from_endpoint(endpoint: str) -> str:
    """``…/v1/chat/completions`` → ``…/v1``."""
    ep = normalize_chat_endpoint(endpoint).rstrip("/")
    if ep.endswith("/chat/completions"):
        return ep[: -len("/chat/completions")]
    if ep.endswith("/v1"):
        return ep
    parsed = urlparse(ep)
    return urlunparse((parsed.scheme, parsed.netloc, "/v1", "", "", ""))


def list_openai_models(
    endpoint: str,
    *,
    api_key: str = "",
    timeout: float = 15.0,
    session: requests.Session | None = None,
) -> list[str]:
    """Liest Modell-IDs von ``GET /v1/models`` (LM Studio / OpenAI-kompatibel)."""
    base = api_base_from_endpoint(endpoint)
    sess = session or requests.Session()
    headers = {"Authorization": f"Bearer {api_key or 'lm-studio'}"}
    resp = sess.get(f"{base}/models", headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    items = data.get("data") or data.get("models") or []
    ids: list[str] = []
    for item in items:
        if isinstance(item, dict) and item.get("id"):
            ids.append(str(item["id"]))
        elif isinstance(item, str):
            ids.append(item)
    return ids


def _is_embedding_model(model_id: str) -> bool:
    mid = model_id.lower()
    return any(k in mid for k in ("embed", "embedding", "nomic-embed"))


def resolve_model_id(
    endpoint: str,
    model: str,
    *,
    api_key: str = "",
    session: requests.Session | None = None,
) -> str:
    """``auto``/leer/``local`` ⇒ erstes Chat-Modell (keine Embeddings), sonst unverändert."""
    m = (model or "").strip()
    if m and m.lower() not in {"auto", "local", "internal-default", "default"}:
        return m
    try:
        ids = list_openai_models(endpoint, api_key=api_key, session=session)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Modell-Autoerkennung fehlgeschlagen: %s — behalte %r", exc, m or "local")
        return m or "local"
    chat_ids = [i for i in ids if not _is_embedding_model(i)]
    pick = chat_ids[0] if chat_ids else (ids[0] if ids else None)
    if not pick:
        return m or "local"
    logger.info("LLM-Modell auto → %s (aus /v1/models)", pick)
    return pick


class BankLLMClient(BaseLLMClient):
    """Client gegen ein OpenAI-kompatibles LLM-Gateway (Bank oder LM Studio)."""

    name = "bank"

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        model: str,
        *,
        timeout: float = 300.0,
        max_retries: int = 2,
        temperature: float = 0.0,
        max_tokens: int = 1_200,
        json_mode: bool | None = None,
        resolve_model: bool = True,
        session: requests.Session | None = None,
    ) -> None:
        if not endpoint:
            raise ValueError(
                "SECPD_LLM_ENDPOINT ist nicht gesetzt — vollständige URL oder "
                "Host wie http://172.16.3.164:1234 angeben."
            )
        self.endpoint = normalize_chat_endpoint(endpoint)
        self.api_key = api_key or "lm-studio"
        self._session = session or requests.Session()
        self.model = (
            resolve_model_id(
                self.endpoint, model, api_key=self.api_key, session=self._session
            )
            if resolve_model
            else (model or "local")
        )
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        # None = versuchen mit json_object, bei 400 ohne erneut
        self.json_mode = json_mode
        safe_model = re.sub(r"[^a-zA-Z0-9._-]+", "_", self.model)[:48]
        self.name = f"bank-{safe_model}" if safe_model else "bank"

    # ------------------------------------------------------------------ #
    def analyze(self, text: str, *, doc_id: str = "") -> TextRiskProfile:
        schema = json.dumps(TextRiskProfile.model_json_schema(), separators=(",", ":"))
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(schema=schema, text=text)},
        ]

        last_error: Exception | None = None
        for attempt in range(2):  # Erstversuch + genau ein Reparatur-Roundtrip
            content = self._chat(messages)
            try:
                return TextRiskProfile.model_validate_json(_strip_fences(content))
            except ValidationError as exc:
                last_error = exc
                logger.warning(
                    "Validierung fehlgeschlagen (doc_id=%s, attempt=%d): %s",
                    doc_id, attempt + 1, exc,
                )
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous answer failed schema validation:\n"
                            f"{exc}\nReturn ONLY the corrected JSON object."
                        ),
                    }
                )
        raise LLMResponseError(f"LLM-Antwort nicht validierbar (doc_id={doc_id}): {last_error}")

    # ------------------------------------------------------------------ #
    def _chat(self, messages: list[dict[str, str]]) -> str:
        prefer_json = True if self.json_mode is None else bool(self.json_mode)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        modes = [True, False] if prefer_json else [False]
        last_exc: Exception | None = None
        # GPT-5.x / neue OpenAI-Modelle: max_completion_tokens statt max_tokens
        use_max_completion = "api.openai.com" in self.endpoint or self.model.startswith(
            ("gpt-5", "o1", "o3", "o4")
        )
        # Manche Reasoning-Modelle akzeptieren temperature=0 nicht
        include_temperature = not use_max_completion

        for json_try in modes:
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
            }
            if include_temperature:
                payload["temperature"] = self.temperature
            if use_max_completion:
                payload["max_completion_tokens"] = self.max_tokens
            else:
                payload["max_tokens"] = self.max_tokens
            if json_try:
                payload["response_format"] = {"type": "json_object"}

            for attempt in range(self.max_retries + 1):
                try:
                    resp = self._session.post(
                        self.endpoint, json=payload, headers=headers, timeout=self.timeout
                    )
                    if resp.status_code == 400:
                        err_txt = (resp.text or "")[:500]
                        if json_try and "response_format" in err_txt.lower():
                            logger.info("Gateway lehnt response_format ab — ohne json_object.")
                            last_exc = requests.HTTPError(f"400: {err_txt}")
                            break
                        if "max_tokens" in err_txt and "max_completion_tokens" not in payload:
                            logger.info("Wechsle auf max_completion_tokens …")
                            payload.pop("max_tokens", None)
                            payload["max_completion_tokens"] = self.max_tokens
                            continue
                        if "temperature" in err_txt.lower() and "temperature" in payload:
                            logger.info("Entferne temperature (Reasoning-Modell) …")
                            payload.pop("temperature", None)
                            continue
                    if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                        wait = 2.0 ** attempt
                        logger.info("Gateway %s — retry in %.0fs", resp.status_code, wait)
                        time.sleep(wait)
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    msg = data["choices"][0]["message"]
                    content = msg.get("content") or ""
                    if not content and msg.get("refusal"):
                        raise LLMResponseError(f"Modell-Refusal: {msg.get('refusal')}")
                    return str(content)
                except requests.RequestException as exc:
                    last_exc = exc
                    if attempt >= self.max_retries:
                        break
                    time.sleep(2.0 ** attempt)
            else:
                continue

        raise LLMResponseError(f"Gateway-Request fehlgeschlagen: {last_exc}")
