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


def env_flag(name: str, default: bool = False) -> bool:
    """Liest eine Ja/Nein-Umgebungsvariable (1/true/yes/on/ja)."""
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "ja"}


def _parse_log_level() -> str:
    raw = (os.getenv("SECPD_LOG_LEVEL") or "").strip().upper()
    aliases = {
        "DEBUG": "DEBUG",
        "INFO": "INFO",
        "WARNING": "WARNING",
        "WARN": "WARNING",
        "ERROR": "ERROR",
        "OFF": "OFF",
        "NONE": "OFF",
        "0": "OFF",
    }
    if raw in aliases:
        return aliases[raw]
    if env_flag("SECPD_DEBUG", False):
        return "INFO"
    return "OFF"


def _parse_keep_screen(log_level: str) -> bool:
    raw = os.getenv("SECPD_DEBUG_KEEP_SCREEN")
    if raw is not None and str(raw).strip() != "":
        return env_flag("SECPD_DEBUG_KEEP_SCREEN", False)
    return log_level != "OFF"


@dataclass(frozen=True)
class Settings:
    """Laufzeit-Konfiguration (immutable Snapshot der ENV beim Aufruf)."""

    llm_mode: str = "mock"                       # SECPD_LLM_MODE: mock | bank | lmstudio
    llm_endpoint: str = ""                       # SECPD_LLM_ENDPOINT (Bank / LM Studio)
    llm_api_key: str = ""                        # SECPD_LLM_API_KEY
    llm_model: str = "internal-default"          # SECPD_LLM_MODEL (lmstudio: auto)
    sec_user_agent: str = ""                     # SECPD_SEC_UA: "Firma name@mail" (EDGAR-Pflicht)
    fmp_api_key: str = ""                        # SECPD_FMP_API_KEY (Ratings-Fetch)
    llm_cache_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "cache" / "llm")
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    models_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "models")
    log_level: str = "OFF"                       # SECPD_LOG_LEVEL / SECPD_DEBUG
    debug_keep_screen: bool = False              # SECPD_DEBUG_KEEP_SCREEN
    llm_allow_mock: bool = True                  # SECPD_LLM_ALLOW_MOCK (Default: ja)
    llm_cache_only: bool = True                  # SECPD_LLM_CACHE_ONLY (Scoring-Default)
    llm_fail_on_miss: bool = False               # SECPD_LLM_FAIL_ON_MISS


def load_settings() -> Settings:
    """Liest die ENV zum Aufrufzeitpunkt (kein Import-Time-Caching)."""
    log_level = _parse_log_level()
    return Settings(
        llm_mode=os.getenv("SECPD_LLM_MODE", "mock"),
        llm_endpoint=os.getenv("SECPD_LLM_ENDPOINT", ""),
        llm_api_key=os.getenv("SECPD_LLM_API_KEY", ""),
        llm_model=os.getenv("SECPD_LLM_MODEL", "internal-default"),
        sec_user_agent=os.getenv("SECPD_SEC_UA", ""),
        fmp_api_key=os.getenv("SECPD_FMP_API_KEY", ""),
        llm_cache_dir=Path(os.getenv("SECPD_LLM_CACHE", PROJECT_ROOT / "data" / "cache" / "llm")),
        log_level=log_level,
        debug_keep_screen=_parse_keep_screen(log_level),
        llm_allow_mock=env_flag("SECPD_LLM_ALLOW_MOCK", True),
        llm_cache_only=env_flag("SECPD_LLM_CACHE_ONLY", True),
        llm_fail_on_miss=env_flag("SECPD_LLM_FAIL_ON_MISS", False),
    )
