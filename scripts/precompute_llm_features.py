#!/usr/bin/env python3
"""Precompute LLM-Textfeatures und füllt den Datei-Cache.

Ideal für OpenAI (schnell) oder LM Studio: einmal alle MD&As klassifizieren
(Cache unter ``data/cache/llm/<namespace>/``), danach ``train.py`` nur noch
Cache-Hits — ohne erneute Inference.

Beispiel (OpenAI / gpt-5.6-luna)::

    # API-Key in start.py → Einstellungen → LLM, oder:
    export SECPD_LLM_MODE=openai
    export SECPD_LLM_API_KEY=sk-...

    python scripts/precompute_llm_features.py \\
      --data data/processed/zenodo_labeled.csv.gz \\
      --min-fyear 2009 --require-financials \\
      --financials data/raw/financials_panel.csv \\
      --llm openai \\
      --out data/processed/llm_features_openai.csv

Unterbrochen? Erneut starten — fertige Cache-Einträge werden übersprungen.
``--llm-refresh`` überschreibt den Cache und bewertet neu.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from secpd.data.zenodo import load_dataset, resolve_columns  # noqa: E402
from secpd.features.textual import extract_text_features, prepare_text  # noqa: E402
from secpd.llm import get_llm_client  # noqa: E402
from secpd.llm.cache import CachedLLMClient  # noqa: E402

logger = logging.getLogger("precompute_llm")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM-Features vorberechnen + cachen")
    p.add_argument("--data", required=True, help="Firm-Year-Datensatz mit MD&A")
    p.add_argument(
        "--llm",
        choices=["mock", "bank", "lmstudio", "openai", "chatgpt"],
        default=None,
        help="Default: SECPD_LLM_MODE (Bulk: openai empfohlen)",
    )
    p.add_argument(
        "--llm-refresh",
        action="store_true",
        help="Cache ignorieren und Texte neu bewerten",
    )
    p.add_argument(
        "--financials",
        default=None,
        help="Optional: Panel für --require-financials",
    )
    p.add_argument("--min-fyear", type=int, default=None)
    p.add_argument("--require-financials", action="store_true")
    p.add_argument("--max-chars", type=int, default=12_000)
    p.add_argument("--sample", type=int, default=None, help="Nur erste N Zeilen (Smoke-Test)")
    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument(
        "--out",
        default=None,
        help="Optionale Feature-Tabelle (.csv/.parquet) zusätzlich zum Cache",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Nur Cache-Hits/Misses zählen (kein LLM-Call)",
    )
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    df = load_dataset(args.data)
    if args.financials:
        panel = pd.read_csv(args.financials)
        panel.columns = [c.lower() for c in panel.columns]
        df = df.merge(panel, on=["cik", "fyear"], how="left", suffixes=("", "_fin"))
    if args.min_fyear is not None and "fyear" in df.columns:
        before = len(df)
        df = df.loc[pd.to_numeric(df["fyear"], errors="coerce") >= int(args.min_fyear)].copy()
        logger.info("min-fyear=%d: %d → %d", args.min_fyear, before, len(df))
    if args.require_financials:
        if "total_assets" not in df.columns:
            logger.error("--require-financials braucht --financials")
            return 2
        before = len(df)
        df = df.loc[df["total_assets"].notna()].copy()
        logger.info("require-financials: %d → %d", before, len(df))
    if args.sample:
        df = df.head(int(args.sample)).copy()

    if not any(c in df.columns for c in ("label", "label_default", "misstate")):
        df["label"] = 0
    cols = resolve_columns(df)
    if cols.text_col is None:
        logger.error("Keine Textspalte (mda/text) gefunden.")
        return 2
    df = df.reset_index(drop=True)
    logger.info("Zeilen für LLM: %d", len(df))

    mode = args.llm
    if mode is None:
        mode = os.environ.get("SECPD_LLM_MODE") or "openai"
    client = get_llm_client(
        mode, cached=True, force_refresh=bool(args.llm_refresh)
    )
    logger.info("LLM-Client: %s (refresh=%s)", client.name, bool(args.llm_refresh))

    if args.dry_run:
        if not isinstance(client, CachedLLMClient):
            logger.error("--dry-run braucht Cache-Wrapper")
            return 2
        hits = misses = 0
        seen: set[str] = set()
        for raw in df[cols.text_col].astype(str):
            prepared = prepare_text(raw, max_chars=args.max_chars)
            if prepared in seen:
                continue
            seen.add(prepared)
            if client._path_for(prepared).exists():  # noqa: SLF001
                hits += 1
            else:
                misses += 1
        logger.info(
            "Cache dry-run (unique Texte): hits=%d misses=%d (%.1f%% hit)",
            hits,
            misses,
            100 * hits / max(1, hits + misses),
        )
        if isinstance(client, CachedLLMClient):
            logger.info("Cache-Verzeichnis: %s", client.cache_dir)
        return 0

    t0 = time.time()
    feats = extract_text_features(
        df,
        client=client,
        text_col=cols.text_col,
        id_col=cols.id_col,
        max_chars=args.max_chars,
        progress_every=args.progress_every,
    )
    elapsed = time.time() - t0
    logger.info(
        "Fertig in %.1f min (%.2f s/doc)",
        elapsed / 60.0,
        elapsed / max(1, len(df)),
    )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower() == ".parquet":
            feats.to_parquet(out_path, index=False)
        else:
            feats.to_csv(out_path, index=False)
        logger.info("Feature-Tabelle: %s (%d Zeilen)", out_path, len(feats))

    if isinstance(client, CachedLLMClient):
        logger.info("Cache-Verzeichnis: %s", client.cache_dir)
        logger.info(
            "Nächstes Training: "
            "SECPD_LLM_MODE=lmstudio SECPD_LLM_ENDPOINT=%s "
            "python train.py … --mode combined --llm lmstudio",
            __import__("os").environ.get("SECPD_LLM_ENDPOINT") or "http://172.16.3.164:1234",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
