"""Deterministischer Mock-Client für die Entwicklung ohne API-Key.

Design-Entscheidung: Statt zufälliger Werte liefert der Mock **deterministische
Heuristiken** (Hedge-Wort-Dichte, Wiederholungs-/Komplexitätsmaße, Risiko-
Lexikon) plus einen hash-geseedeten Mini-Jitter. Vorteile:

* Reproduzierbar — gleicher Text ⇒ identisches Profil (testbar, cachebar).
* Plausible Varianz — die Downstream-Pipeline lässt sich sinnvoll trainieren
  und der Ensemble-Pfad realistisch entwickeln.
* Gratis-Baseline — auf dem Bank-Server lässt sich prüfen, ob das echte LLM
  gegenüber trivialen lexikalischen Statistiken überhaupt Mehrwert liefert.

Nur Python-Standardbibliothek, keine externen Abhängigkeiten.
"""
from __future__ import annotations

import hashlib
import random
import re
from collections import Counter

from .base import BaseLLMClient
from .schema import TextRiskProfile

# --------------------------------------------------------------------------- #
# Lexika (bewusst kompakt; Erweiterung jederzeit möglich)
# --------------------------------------------------------------------------- #
HEDGE_WORDS: frozenset[str] = frozenset(
    """
    may might could would should possibly potentially approximately
    believe believes believed anticipate anticipates expect expects
    estimate estimates estimated assume assumes assumption assumptions
    uncertain uncertainty uncertainties likely unlikely appears appear
    seems generally substantially significantly subject contingent
    approximately roughly certain various
    """.split()
)

NEGATIVE_RISK_TERMS: frozenset[str] = frozenset(
    """
    impairment impairments default defaults litigation lawsuit lawsuits
    restatement restatements deficiency deficiencies weakness weaknesses
    covenant covenants breach breaches decline declines declined loss losses
    writedown writedowns write-off write-offs delinquency delinquencies
    downgrade downgraded insolvency bankruptcy fraud investigation
    investigations subpoena penalty penalties fines adverse adversely
    volatility restructuring layoffs going-concern concern doubt
    """.split()
)

POSITIVE_TERMS: frozenset[str] = frozenset(
    """
    growth grow strong stronger record improved improvement improvements
    profitable profitability exceeded robust momentum favorable favorably
    innovation expanding expansion successful success confident
    """.split()
)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_TOKEN = re.compile(r"[A-Za-z][A-Za-z\-']*")


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


class MockLLMClient(BaseLLMClient):
    """Heuristischer Offline-Ersatz für den Bank-LLM-Client."""

    name = "mock-heuristic-v1"

    def __init__(self, jitter: float = 0.04) -> None:
        self.jitter = float(jitter)

    # ------------------------------------------------------------------ #
    def analyze(self, text: str, *, doc_id: str = "") -> TextRiskProfile:
        toks = _tokens(text)
        if len(toks) < 10:
            return TextRiskProfile.fallback(reason="text too short for analysis")

        sentences = [s for s in _SENT_SPLIT.split(text) if s.strip()]
        n_tok = len(toks)
        n_sent = max(len(sentences), 1)

        # Deterministischer Jitter, geseeded über den Text-Hash.
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
        rng = random.Random(seed)
        j = lambda: rng.uniform(-self.jitter, self.jitter)  # noqa: E731

        # 1) Vagueness: Hedge-Wörter pro 100 Tokens (typ. MD&A ~1–5).
        hedge_count = sum(1 for t in toks if t in HEDGE_WORDS)
        hedge_density = 100.0 * hedge_count / n_tok
        vagueness = _clamp(hedge_density / 5.0 + j())

        # 2) Redundancy: wiederholte Trigramme + geringe Token-Vielfalt.
        trigrams = list(zip(toks, toks[1:], toks[2:]))
        if trigrams:
            repeated_share = 1.0 - len(set(trigrams)) / len(trigrams)
        else:
            repeated_share = 0.0
        unique_ratio = len(set(toks)) / n_tok
        redundancy = _clamp(3.0 * repeated_share + 0.5 * (1.0 - unique_ratio) + j())

        # 3) Complexity: Satzlänge + Anteil langer Wörter.
        avg_sent_len = n_tok / n_sent
        long_word_share = sum(1 for t in toks if len(t) >= 9) / n_tok
        complexity = _clamp(
            0.6 * _clamp((avg_sent_len - 10.0) / 25.0) + 0.4 * _clamp(4.0 * long_word_share) + j()
        )

        # 4) Risk-Sentiment: Lexikon-Bilanz, geglättet.
        neg_hits = Counter(t for t in toks if t in NEGATIVE_RISK_TERMS)
        pos_hits = Counter(t for t in toks if t in POSITIVE_TERMS)
        neg, pos = sum(neg_hits.values()), sum(pos_hits.values())
        sentiment = _clamp((pos - neg) / (pos + neg + 3.0) + j(), lo=-1.0, hi=1.0)

        top_neg = [term for term, _ in neg_hits.most_common(5)]
        summary = (
            f"[MOCK] {n_tok} Tokens, {n_sent} Sätze; Hedge-Dichte "
            f"{hedge_density:.1f}/100; dominierende Risikobegriffe: "
            f"{', '.join(top_neg) if top_neg else 'keine'}."
        )

        return TextRiskProfile(
            vagueness_score=round(vagueness, 4),
            redundancy_score=round(redundancy, 4),
            complexity_score=round(complexity, 4),
            risk_sentiment=round(sentiment, 4),
            confidence=0.35,  # Heuristik ⇒ bewusst niedrige Konfidenz
            obfuscation_indicators=top_neg,
            risk_summary=summary,
        )
