"""Structured-Output-Schema für die LLM-Textanalyse.

Dieses Pydantic-Modell ist der zentrale Vertrag zwischen LLM-Schicht und
ML-Pipeline: Jeder Client (Mock oder Bank) MUSS ein valides
:class:`TextRiskProfile` liefern. Die numerischen Felder werden über
:meth:`TextRiskProfile.to_features` zu Feature-Spalten projiziert.
"""
from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_SUMMARY_CHARS = 600
MAX_INDICATORS = 8


class TextRiskProfile(BaseModel):
    """LLM-Bewertung eines Filing-Textabschnitts (z. B. MD&A eines 10-K)."""

    model_config = ConfigDict(extra="ignore")  # robust gegen zusätzliche LLM-Felder

    vagueness_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Grad an Vagheit/Hedging (0 = präzise, quantifiziert; "
            "1 = maximal ausweichend, viele Modalverben und unbestimmte Aussagen)."
        ),
    )
    redundancy_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Anteil repetitiver/boilerplate-artiger Passagen (0 = keine, 1 = extrem).",
    )
    complexity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Sprachliche Komplexität (Satzlänge, Verschachtelung, Jargon-Dichte).",
    )
    risk_sentiment: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Risikoton des Texts (-1 = stark negativ/krisenhaft, +1 = zuversichtlich).",
    )
    confidence: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Selbsteinschätzung des Modells zur Verlässlichkeit dieser Bewertung.",
    )
    obfuscation_indicators: list[str] = Field(
        default_factory=list,
        description="Kurze Stichworte zu auffälligen Verschleierungs-/Risikosignalen im Text.",
    )
    risk_summary: str = Field(
        "",
        description="Knappe (<= 3 Sätze) Risiko-Zusammenfassung in natürlicher Sprache.",
    )

    #: Felder, die als numerische ML-Features exportiert werden.
    FEATURE_FIELDS: ClassVar[tuple[str, ...]] = (
        "vagueness_score",
        "redundancy_score",
        "complexity_score",
        "risk_sentiment",
        "confidence",
    )

    # ------------------------------------------------------------------ #
    # Robustheit: lieber clippen als an Längen-Constraints scheitern.
    # ------------------------------------------------------------------ #
    @field_validator("risk_summary", mode="before")
    @classmethod
    def _clip_summary(cls, v: Any) -> Any:
        if isinstance(v, str) and len(v) > MAX_SUMMARY_CHARS:
            return v[: MAX_SUMMARY_CHARS - 1] + "…"
        return v

    @field_validator("obfuscation_indicators", mode="before")
    @classmethod
    def _cap_indicators(cls, v: Any) -> Any:
        if isinstance(v, list):
            return [str(x)[:80] for x in v[:MAX_INDICATORS]]
        return v

    # ------------------------------------------------------------------ #
    # Projektion in den Feature-Raum der sklearn-Pipeline.
    # ------------------------------------------------------------------ #
    @classmethod
    def feature_names(cls, prefix: str = "llm_") -> list[str]:
        """Spaltennamen der numerischen Features (für Pipeline-Definitionen)."""
        return [f"{prefix}{f}" for f in cls.FEATURE_FIELDS]

    def to_features(self, prefix: str = "llm_") -> dict[str, float]:
        """Numerische Felder als flaches Dict (eine DataFrame-Zeile)."""
        return {f"{prefix}{f}": float(getattr(self, f)) for f in self.FEATURE_FIELDS}

    @classmethod
    def fallback(cls, reason: str = "unavailable") -> "TextRiskProfile":
        """Neutrales Profil für Fehlerfälle (Batch-Jobs sollen nicht abbrechen).

        ``confidence=0.0`` markiert das Profil eindeutig als Fallback und ist
        dadurch auch downstream als Feature interpretierbar.
        """
        return cls(
            vagueness_score=0.5,
            redundancy_score=0.5,
            complexity_score=0.5,
            risk_sentiment=0.0,
            confidence=0.0,
            obfuscation_indicators=[],
            risk_summary=f"[fallback] {reason}",
        )
