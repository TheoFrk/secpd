#!/usr/bin/env python3
"""Kurzer Erreichbarkeits-Check für LM Studio / OpenAI-kompatible Gateways.

Beispiel::

    python scripts/ping_llm.py --endpoint http://172.16.3.164:1234
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import requests  # noqa: E402

from secpd.llm.bank import (  # noqa: E402
    DEFAULT_LMSTUDIO_HOST,
    list_openai_models,
    normalize_chat_endpoint,
)
from secpd.llm.schema import TextRiskProfile  # noqa: E402
from secpd.llm import get_llm_client  # noqa: E402
import os  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ping LLM Gateway / LM Studio")
    p.add_argument("--endpoint", default=os.getenv("SECPD_LLM_ENDPOINT") or DEFAULT_LMSTUDIO_HOST)
    p.add_argument("--model", default=os.getenv("SECPD_LLM_MODEL") or "auto")
    p.add_argument("--api-key", default=os.getenv("SECPD_LLM_API_KEY") or "lm-studio")
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--analyze", action="store_true", help="Mini-TextRiskProfile-Call")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    chat = normalize_chat_endpoint(args.endpoint)
    print(f"Endpoint (chat): {chat}")
    try:
        models = list_openai_models(args.endpoint, api_key=args.api_key, timeout=args.timeout)
        print(f"Modelle ({len(models)}):")
        for m in models[:20]:
            print(f"  · {m}")
        if not models:
            print("  (keine — in LM Studio ein Modell laden / Server starten)")
            return 2
    except requests.RequestException as exc:
        print(f"FEHLER: /v1/models nicht erreichbar: {exc}")
        print("Hinweis: LM Studio → Local Server starten, CORS/Bind auf 0.0.0.0 prüfen.")
        return 2

    if args.analyze:
        os.environ["SECPD_LLM_MODE"] = "lmstudio"
        os.environ["SECPD_LLM_ENDPOINT"] = args.endpoint
        os.environ["SECPD_LLM_MODEL"] = args.model
        os.environ["SECPD_LLM_API_KEY"] = args.api_key
        os.environ["SECPD_LLM_TIMEOUT"] = str(args.timeout)
        client = get_llm_client("lmstudio", cached=True)
        print(f"Client: {client.name}")
        sample = (
            "Liquidity remains adequate. However, we may face uncertainty regarding "
            "covenant compliance if operating cash flows deteriorate further."
        )
        profile = client.analyze(sample, doc_id="ping")
        assert isinstance(profile, TextRiskProfile)
        print("Analyze OK:")
        print(f"  vagueness={profile.vagueness_score:.2f}  "
              f"sentiment={profile.risk_sentiment:.2f}  "
              f"confidence={profile.confidence:.2f}")
        print(f"  summary: {profile.risk_summary[:160]}")
    else:
        print("OK — Server antwortet. Mit --analyze einen echten Klassifikations-Call testen.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
