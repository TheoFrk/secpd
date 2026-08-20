#!/usr/bin/env python3
"""Interaktive Start-Oberfläche für SEC-PD.

Bedienung::

    source .venv/bin/activate
    python start.py

Menü: Unternehmen scoren, Modellgüte, Hilfe, Einstellungen, Beenden.
Die Logik liegt in ``src/secpd/cli/`` (ui, catalog, scoring, quality, settings).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def _bootstrap() -> None:
    try:
        import pandas  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError:
        print()
        print("  Pandas/sklearn fehlen. Bitte zuerst die virtuelle Umgebung aktivieren:")
        print()
        print("      cd", ROOT)
        print("      source .venv/bin/activate")
        print("      python start.py")
        print()
        raise SystemExit(1)


if __name__ == "__main__":
    _bootstrap()
    from secpd.cli.app import main

    main()
