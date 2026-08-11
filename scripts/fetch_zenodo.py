#!/usr/bin/env python3
"""Lädt die Zenodo-Dateien (Record 17121948) mit MD5-Prüfung. Nur Home-Setup.

Beispiel:
    python scripts/fetch_zenodo.py --files aaer_mark5.csv firm_years_labels.json
"""
from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path

import requests

RECORD_API = "https://zenodo.org/api/records/17121948"
CHUNK = 1 << 20  # 1 MiB


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    p = argparse.ArgumentParser(description="Zenodo-Downloader (Record 17121948)")
    p.add_argument("--files", nargs="+", default=["aaer_mark5.csv"],
                   help="Dateinamen aus dem Record (firm_years_labels.json ist ~716 MB!)")
    p.add_argument("--dest", default="data/raw")
    args = p.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    meta = requests.get(RECORD_API, timeout=60).json()
    index = {f["key"]: f for f in meta.get("files", [])}

    for name in args.files:
        if name not in index:
            logging.error("Datei %r nicht im Record. Verfügbar: %s", name, sorted(index))
            return 2
        entry = index[name]
        url = entry["links"]["self"]
        expected = str(entry.get("checksum", "")).removeprefix("md5:")
        target = dest / name
        logging.info("Lade %s (%.1f MB) …", name, entry.get("size", 0) / 1e6)
        with requests.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with target.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=CHUNK):
                    fh.write(chunk)
        actual = md5_of(target)
        if expected and actual != expected:
            logging.error("MD5-Mismatch bei %s: %s != %s", name, actual, expected)
            return 3
        logging.info("OK: %s (md5 verifiziert)", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
