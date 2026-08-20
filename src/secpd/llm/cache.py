"""Transparenter Datei-Cache für LLM-Antworten (Decorator-Pattern).

Motivation im 1-Wochen-Sprint:

* Kein doppeltes Feuern teurer/langsamer LLM-Calls bei Re-Runs.
* Der Cache (JSON pro Text-Hash) ist **committbar**: Einmal auf dem
  Bank-Server berechnete Profile lassen sich zu Hause exakt „replayen" —
  und umgekehrt lassen sich Mock-Profile deterministisch teilen.
* Namespaces trennen Mock-/Bank-/Modell-Varianten sauber.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from .base import BaseLLMClient
from .schema import TextRiskProfile

logger = logging.getLogger(__name__)


class CachedLLMClient(BaseLLMClient):
    """Wrappt einen beliebigen Client und cached Profile als JSON-Dateien."""

    def __init__(
        self,
        inner: BaseLLMClient,
        cache_dir: Path | str,
        *,
        force_refresh: bool = False,
        cache_only: bool = False,
    ) -> None:
        self.inner = inner
        self.cache_dir = Path(cache_dir) / inner.cache_namespace
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.force_refresh = bool(force_refresh)
        self.cache_only = bool(cache_only) and not self.force_refresh
        self._cache_hits = 0
        self._cache_misses = 0
        self.name = f"cached({inner.name})"
        if self.force_refresh:
            self.name += "+refresh"
        elif self.cache_only:
            self.name += "+cache-only"

    @property
    def cache_namespace(self) -> str:
        return self.inner.cache_namespace

    def _path_for(self, text: str) -> Path:
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
        return self.cache_dir / f"{key}.json"

    def analyze(self, text: str, *, doc_id: str = "") -> TextRiskProfile:
        path = self._path_for(text)
        if path.exists() and not self.force_refresh:
            try:
                profile = TextRiskProfile.model_validate_json(path.read_text(encoding="utf-8"))
                self._cache_hits += 1
                return profile
            except Exception:  # noqa: BLE001 — korrupten Cache-Eintrag neu berechnen
                logger.warning("Korrupter Cache-Eintrag wird neu berechnet: %s", path.name)

        self._cache_misses += 1
        n_miss = self._cache_misses
        if self.cache_only:
            if n_miss <= 3 or n_miss % 10_000 == 0:
                logger.info("Cache-Miss (cache-only, kein LLM-Call) #%d: %s", n_miss, path.name)
            return TextRiskProfile.fallback(reason="cache miss")

        if n_miss <= 3 or n_miss % 10_000 == 0:
            logger.info("Cache-Miss #%d → LLM %s: %s", n_miss, self.inner.name, path.name)
        profile = self.inner.analyze(text, doc_id=doc_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(path)  # atomarer Write
        return profile
