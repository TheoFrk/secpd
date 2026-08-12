"""Zentrale Konfiguration — ausschließlich über Umgebungsvariablen.

Alle Pfade sind relativ zum Projekt-Root aufgelöst, damit das Repo als
Ganzes (inkl. Caches und Assets) zwischen Home-PC und Bank-Server per Git
transferiert werden kann.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """Laufzeit-Konfiguration (immutable Snapshot der ENV beim Aufruf)."""

    llm_mode: str = "mock"                       # SECPD_LLM_MODE: mock | bank | lmstudio
    llm_endpoint: str = ""                       # SECPD_LLM_ENDPOINT (Bank / LM Studio)
    llm_api_key: str = ""                        # SECPD_LLM_API_KEY
    llm_model: str = "internal-default"          # SECPD_LLM_MODEL (lmstudio: auto)
    sec_user_agent: str = ""                     # SECPD_SEC_UA: "Firma name@mail" (EDGAR-Pflicht)
    llm_cache_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "cache" / "llm")
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    models_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "models")


def load_settings() -> Settings:
    """Liest die ENV zum Aufrufzeitpunkt (kein Import-Time-Caching)."""
    return Settings(
        llm_mode=os.getenv("SECPD_LLM_MODE", "mock"),
        llm_endpoint=os.getenv("SECPD_LLM_ENDPOINT", ""),
        llm_api_key=os.getenv("SECPD_LLM_API_KEY", ""),
        llm_model=os.getenv("SECPD_LLM_MODEL", "internal-default"),
        sec_user_agent=os.getenv("SECPD_SEC_UA", ""),
        llm_cache_dir=Path(os.getenv("SECPD_LLM_CACHE", PROJECT_ROOT / "data" / "cache" / "llm")),
    )
