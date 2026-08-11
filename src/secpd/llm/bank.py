"""Echter LLM-Client für den Bank-Server (air-gapped GCP Workstation).

Annahme: Das interne Gateway ist OpenAI-kompatibel (POST /chat/completions).
Alle Stellen, die an das konkrete Commerzbank-Gateway anzupassen sind, sind
mit ``# ADAPT:`` markiert. Konfiguration ausschließlich über ENV:

* ``SECPD_LLM_ENDPOINT`` — vollständige URL des Completions-Endpoints
* ``SECPD_LLM_API_KEY``  — interner Key (nie ins Repo committen!)
* ``SECPD_LLM_MODEL``    — Modellname des Gateways

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

import requests
from pydantic import ValidationError

from .base import BaseLLMClient, LLMResponseError
from .schema import TextRiskProfile

logger = logging.getLogger(__name__)

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

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


class BankLLMClient(BaseLLMClient):
    """Client gegen das interne, OpenAI-kompatible LLM-Gateway."""

    name = "bank"

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        model: str,
        *,
        timeout: float = 60.0,
        max_retries: int = 2,
        temperature: float = 0.0,
        session: requests.Session | None = None,
    ) -> None:
        if not endpoint:
            raise ValueError(
                "SECPD_LLM_ENDPOINT ist nicht gesetzt — auf dem Bank-Server "
                "erforderlich (siehe README, Abschnitt ENV-Variablen)."
            )
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self._session = session or requests.Session()
        # Modellname in den Cache-Namespace aufnehmen (Modellwechsel ⇒ frischer Cache).
        self.name = f"bank-{model}" if model else "bank"

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
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": 800,
            # ADAPT: entfernen, falls das Gateway response_format nicht kennt.
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Content-Type": "application/json",
            # ADAPT: Auth-Schema des internen Gateways (Bearer/Api-Key/Header-Name).
            "Authorization": f"Bearer {self.api_key}",
        }

        for attempt in range(self.max_retries + 1):
            try:
                resp = self._session.post(
                    self.endpoint, json=payload, headers=headers, timeout=self.timeout
                )
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    wait = 2.0 ** attempt
                    logger.info("Gateway %s — retry in %.0fs", resp.status_code, wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                # ADAPT: Response-Pfad, falls das Gateway ein anderes Format liefert.
                return str(data["choices"][0]["message"]["content"])
            except requests.RequestException as exc:
                if attempt >= self.max_retries:
                    raise LLMResponseError(f"Gateway-Request fehlgeschlagen: {exc}") from exc
                time.sleep(2.0 ** attempt)
        raise LLMResponseError("Gateway-Request fehlgeschlagen (Retries erschöpft).")
