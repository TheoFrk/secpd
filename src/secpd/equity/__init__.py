"""Aktienwert-Modul: SEC-PD als Overlay für die Bachelor-Praxis (S1–S4).

Die Bachelorarbeit bewertet europäische Aktien über Preis (S1), Lexikon (S2),
FinBERT (S3) und News-LLM (S4). Dieses Paket liefert Szenario S5: einen
langsam laufenden 10-K-Distress-Score, der als Qualitäts-/Aktienwert-Signal
auf Daily-Panels gejoint werden kann (PIT über ``filing_date``).
"""

from .overlay import attach_equity_scores, pd_to_equity_quality

__all__ = ["attach_equity_scores", "pd_to_equity_quality"]
