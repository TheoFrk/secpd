"""Hauptmenü der interaktiven SEC-PD-Oberfläche."""
from __future__ import annotations

import os

import pandas as pd

from secpd.cli import state
from secpd.cli.catalog import (
    active_model_meta,
    active_model_path,
    list_model_catalog,
    warn_model_coherence,
)
from secpd.cli.paths import LABELED, PANEL, ROOT
from secpd.cli.quality import show_model_quality
from secpd.cli.scoring import flow_score_company, load_company_index
from secpd.cli.settings import load_secrets_env, settings_menu
from secpd.cli.ui import C, ask, banner, clear, horizon_label, hr, pause


def require_files() -> list[str]:
    missing = []
    if not LABELED.exists():
        missing.append(str(LABELED.relative_to(ROOT)))
    if not PANEL.exists():
        missing.append(str(PANEL.relative_to(ROOT)))
    if not list_model_catalog():
        missing.append("models/*.joblib (bitte zuerst trainieren)")
    return missing


def show_help() -> None:
    clear()
    banner()
    print(f"  {C.BOLD}Hilfe{C.RESET}")
    hr()
    print(
        """
  1) 10-K scoren
     Strukturierte Eingabe:
       1  Label-Set (Firma/CIK suchen)
       2  Datei laden (CSV/JSON oder MD&A-Text)
       3  EDGAR live (Finanzen + optional 8-Ks)
       4  Fragebogen (Kennzahlen + MD&A manuell)
     → 10-K wählen → Prognosehorizont in Monaten
       (12 / 24 / 36 / 60 / 120 oder frei, z. B. 18)
     → Rating + PD + Termstruktur-Vorausschau
     CSV-Export optional (inkl. pd_12m … pd_120m)

  2) Modellgüte
     · Rating: MAE / Spearman / ±1-Notch
     · Default/Fraud: ROC-AUC / PR-AUC / Brier
     · Fraud-Bundles sind als experimentell markiert

  4) Einstellungen
     · Vorausschauhorizont, LLM, SEC-UA, Zenodo, EDGAR, Ratings (NRSRO), Training
     · Training mit eigenem --default-horizon (echte Labels)

  Hinweise
     · Primär: Shadow-Rating (ordinal, Agency-Skala)
     · PD sekundär, Horizont beim Scoren wählbar
     · 12 / 24 / 36 M: natives Default-Modell, sonst Hazard-Termstruktur
     · Echte 5J/10J-PD: neu trainieren mit Horizont 60/120
     · Fraud (AAER) ist experimentell — wenige Positive, schlechter Skill
"""
    )
    pause()


def main_menu() -> None:
    while True:
        missing = require_files()
        index = load_company_index() if LABELED.exists() else pd.DataFrame(columns=["cik", "name"])
        meta = active_model_meta()

        clear()
        banner()
        if missing:
            print(f"  {C.YELLOW}Fehlende Dateien:{C.RESET}")
            for m in missing:
                print(f"    · {m}")
            print(f"  {C.DIM}Über Einstellungen (4) nachladen / trainieren.{C.RESET}")
            print()

        n_co = index["cik"].nunique() if not index.empty else 0
        label = meta.get("label_source") or "?"
        active = active_model_path()
        active_s = active.name if active else "fehlt"
        print(
            f"  {C.DIM}Label-Set: {n_co} Unternehmen · aktiv: {active_s} · "
            f"target={label} · LLM={os.environ.get('SECPD_LLM_MODE', 'mock')}{C.RESET}"
        )
        warn_model_coherence()
        print()
        if label == "default":
            h = state.FORECAST_HORIZON_MONTHS or int(meta.get("default_horizon_months") or 12)
            print(f"  {C.CYAN}1{C.RESET}  10-K scoren (PD, Horizont wählbar)")
            print(f"     {C.DIM}Session-Vorausschau: {horizon_label(h)}{C.RESET}")
        elif label == "rating":
            h = state.FORECAST_HORIZON_MONTHS or 12
            print(f"  {C.CYAN}1{C.RESET}  10-K scoren (Rating + PD, Horizont wählbar)")
            print(f"     {C.DIM}Session-Vorausschau: {horizon_label(h)}{C.RESET}")
        else:
            print(f"  {C.CYAN}1{C.RESET}  10-K scoren")
        print(f"  {C.CYAN}2{C.RESET}  Modellgüte anzeigen")
        print(f"  {C.CYAN}3{C.RESET}  Hilfe")
        print(f"  {C.CYAN}4{C.RESET}  Einstellungen")
        print(f"  {C.CYAN}0{C.RESET}  Beenden")
        print()
        choice = ask("Auswahl", "1")

        if choice == "1":
            if not list_model_catalog():
                print(f"  {C.RED}Kein trainiertes Modell — Einstellungen → Trainieren.{C.RESET}")
                pause()
                continue
            flow_score_company(index)
        elif choice == "2":
            show_model_quality()
        elif choice == "3":
            show_help()
        elif choice == "4":
            settings_menu()
            active_model_meta(refresh=True)
        elif choice in {"0", "q", "quit", "exit"}:
            print(f"\n  {C.DIM}Tschüss.{C.RESET}\n")
            return
        else:
            print(f"  {C.YELLOW}Bitte 0–4 wählen.{C.RESET}")
            pause()


def main() -> None:
    load_secrets_env()
    main_menu()
