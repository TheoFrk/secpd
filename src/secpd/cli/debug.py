"""Debug-Flags: Terminal-Logs, Mock-Sperre, Cache-Verhalten."""
from __future__ import annotations

import logging
import sys

from secpd.config import load_settings
from secpd.llm.base import BaseLLMClient

_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}


def keep_screen() -> bool:
    """True: Menüs nicht löschen, damit Logs im Terminal bleiben."""
    return load_settings().debug_keep_screen


def format_debug_status() -> str:
    """Eine Zeile für Menüs (logs=INFO · mock=nein · cache-only)."""
    s = load_settings()
    mock = "ja" if s.llm_allow_mock else "nein"
    cache = "cache-only" if s.llm_cache_only else "API-bei-Miss"
    miss = "Miss=Abbruch" if s.llm_fail_on_miss else "Miss=Fallback"
    return f"logs={s.log_level} · mock={mock} · {cache} · {miss}"


def configure_logging() -> None:
    """Hängt einen stderr-Handler an, sobald Logs nicht OFF sind."""
    settings = load_settings()
    root = logging.getLogger()
    ours = [h for h in root.handlers if getattr(h, "_secpd_debug", False)]

    if settings.log_level == "OFF":
        for h in ours:
            root.removeHandler(h)
            h.close()
        return

    level = _LEVELS.get(settings.log_level, logging.INFO)
    root.setLevel(level)
    if not ours:
        handler = logging.StreamHandler(sys.stderr)
        handler._secpd_debug = True  # type: ignore[attr-defined]
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        root.addHandler(handler)
        ours = [handler]
    for h in ours:
        h.setLevel(level)


def reset_logging_for_tests() -> None:
    """Test-Cleanup: von uns gesetzte Handler entfernen."""
    root = logging.getLogger()
    for h in list(root.handlers):
        if getattr(h, "_secpd_debug", False):
            root.removeHandler(h)
            h.close()
    root.setLevel(logging.WARNING)


def llm_client_label(client: BaseLLMClient) -> str:
    cache_dir = getattr(client, "cache_dir", None)
    ns = cache_dir.name if cache_dir is not None else client.name
    return f"{client.name}, cache={ns}"


def print_llm_cache_report(client: BaseLLMClient) -> None:
    """Treffer/Fehltreffer nach der Textanalyse (auch ohne DEBUG sichtbar)."""
    from secpd.cli.ui import C

    hits = int(getattr(client, "_cache_hits", 0) or 0)
    misses = int(getattr(client, "_cache_misses", 0) or 0)
    cache_dir = getattr(client, "cache_dir", None)
    where = str(cache_dir) if cache_dir is not None else client.name
    print(f"  {C.DIM}LLM-Cache {where}: {hits} Hits, {misses} Misses{C.RESET}")
    if misses and getattr(client, "cache_only", False):
        print(
            f"  {C.YELLOW}Cache-Miss → kein GPT-Call (cache-only), "
            f"neutrales Fallback-Profil.{C.RESET}"
        )


def raise_if_cache_miss_forbidden(client: BaseLLMClient) -> None:
    settings = load_settings()
    misses = int(getattr(client, "_cache_misses", 0) or 0)
    if settings.llm_fail_on_miss and misses:
        cache_dir = getattr(client, "cache_dir", None)
        raise RuntimeError(
            f"{misses} LLM-Cache-Miss(es) unter {cache_dir or client.name} — "
            "Abbruch (SECPD_LLM_FAIL_ON_MISS=1). Cache füllen oder "
            "Einstellungen → Debug → Cache-only aus."
        )
