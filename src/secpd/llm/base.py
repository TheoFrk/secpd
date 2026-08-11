"""Abstraktes Client-Interface für die LLM-Textanalyse.

Home-Setup und Bank-Server implementieren dasselbe Interface; der Austausch
erfolgt ausschließlich über die Factory (``get_llm_client``) bzw. die
Umgebungsvariable ``SECPD_LLM_MODE`` — der Pipeline-Code bleibt identisch.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Literal, Sequence

from .schema import TextRiskProfile

logger = logging.getLogger(__name__)


class LLMResponseError(RuntimeError):
    """LLM-Antwort konnte auch nach Reparaturversuch nicht validiert werden."""


class BaseLLMClient(ABC):
    """Gemeinsames Interface aller LLM-Clients (Mock, Bank, Cache-Wrapper)."""

    #: Kurzname, u. a. Bestandteil des Cache-Namespaces.
    name: str = "base"

    @property
    def cache_namespace(self) -> str:
        """Namespace für den Datei-Cache (Mock- und Bank-Ergebnisse getrennt)."""
        return self.name

    @abstractmethod
    def analyze(self, text: str, *, doc_id: str = "") -> TextRiskProfile:
        """Analysiert einen Textabschnitt und liefert ein valides Profil.

        Parameters
        ----------
        text:
            Vorverarbeiteter Filing-Ausschnitt (siehe ``features.textual.prepare_text``).
        doc_id:
            Optionale Dokument-ID, nur für Logging/Nachvollziehbarkeit.
        """

    def analyze_batch(
        self,
        texts: Sequence[str],
        *,
        doc_ids: Sequence[str] | None = None,
        on_error: Literal["raise", "fallback"] = "fallback",
    ) -> list[TextRiskProfile]:
        """Sequenzielle Batch-Analyse mit optionalem Fallback pro Dokument."""
        ids = list(doc_ids) if doc_ids is not None else [""] * len(texts)
        results: list[TextRiskProfile] = []
        for text, doc_id in zip(texts, ids):
            try:
                results.append(self.analyze(text, doc_id=doc_id))
            except Exception as exc:  # noqa: BLE001 — Batch soll robust durchlaufen
                if on_error == "raise":
                    raise
                logger.warning("LLM-Analyse fehlgeschlagen (doc_id=%s): %s", doc_id, exc)
                results.append(TextRiskProfile.fallback(reason=str(exc)[:120]))
        return results
