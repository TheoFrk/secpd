"""Synthetischer Datensatz im kanonischen Schema (für Demos, Tests, CI).

Ein latenter Risikofaktor ``u`` treibt sowohl die Finanzkennzahlen (höherer
Leverage, schwächere Liquidität/Profitabilität) als auch den Textstil
(mehr Hedging, mehr negative Risikobegriffe) und die Ausfall-/Fraud-
Wahrscheinlichkeit. Damit fließt nachweislich Signal durch beide Pfade der
Pipeline — der End-to-End-Smoke-Test wird aussagekräftig statt trivial.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NEUTRAL_SENTENCES = [
    "Revenues are generated from product sales through direct channels and distributors.",
    "The Company operates manufacturing facilities in several regions.",
    "Cost of revenues consists primarily of materials, labor and overhead.",
    "Selling, general and administrative expenses include personnel and marketing costs.",
    "The Company's fiscal year ends on December 31.",
    "Capital expenditures relate mainly to equipment and information systems.",
    "The Company maintains insurance coverage customary for its industry.",
]

HEDGY_SENTENCES = [
    "Management believes results may vary substantially due to various uncertainties.",
    "The Company anticipates that certain estimates could potentially require adjustment.",
    "Actual outcomes might differ materially from assumptions, which are subject to uncertainty.",
    "We expect, although it is uncertain, that liquidity would generally be sufficient.",
    "Certain contingent obligations may possibly affect future periods.",
]

NEGATIVE_SENTENCES = [
    "The Company recorded an impairment charge and identified a material weakness in internal control.",
    "A covenant breach occurred and litigation related to the restatement is pending.",
    "There is substantial doubt about the Company's ability to continue as a going concern.",
    "The investigation resulted in penalties, and losses from writedowns increased.",
    "Delinquency rates rose and a downgrade of the credit facility followed.",
]

POSITIVE_SENTENCES = [
    "The Company achieved record revenues driven by strong growth and improved margins.",
    "Robust momentum and successful expansion supported profitability.",
    "Management is confident in the favorable outlook and continued improvement.",
]


def _compose_text(rng: np.random.Generator, u: float, n_sentences: int = 26) -> str:
    """Mischt Satz-Pools; riskantere Firmen (hohes ``u``) klingen vager/negativer."""
    w_neutral = max(0.15, 0.75 - 0.5 * u)
    w_hedgy = 0.15 + 0.35 * u
    w_negative = 0.03 + 0.4 * u
    w_positive = max(0.02, 0.25 - 0.2 * u)
    probs = np.array([w_neutral, w_hedgy, w_negative, w_positive])
    probs = probs / probs.sum()
    pools = [NEUTRAL_SENTENCES, HEDGY_SENTENCES, NEGATIVE_SENTENCES, POSITIVE_SENTENCES]
    sentences = [
        pools[k][rng.integers(len(pools[k]))]
        for k in rng.choice(4, size=n_sentences, p=probs)
    ]
    return " ".join(sentences)


def make_synthetic_dataset(
    n: int = 1_200,
    *,
    seed: int = 42,
    base_rate_logit: float = -2.9,
    year_range: tuple[int, int] = (2015, 2023),
) -> pd.DataFrame:
    """Erzeugt ``n`` Firm-Years mit kanonischen Finanzspalten, ``mda`` und ``label``."""
    rng = np.random.default_rng(seed)

    u = rng.beta(2.0, 5.0, size=n)  # latentes Risiko in [0,1], rechtsschief
    cik = rng.integers(100_000, 999_999, size=n)
    fyear = rng.integers(year_range[0], year_range[1] + 1, size=n)

    total_assets = np.exp(rng.normal(19.0, 1.6, size=n))  # ~ e8 .. e10 USD
    leverage = np.clip(rng.normal(0.45 + 0.35 * u, 0.10), 0.05, 0.98)
    total_liabilities = leverage * total_assets
    equity = total_assets - total_liabilities

    current_assets = total_assets * np.clip(rng.normal(0.42, 0.08, size=n), 0.1, 0.8)
    current_liabilities = current_assets / np.clip(
        rng.normal(1.9 - 1.0 * u, 0.30), 0.5, 4.0
    )
    cash = current_assets * np.clip(rng.normal(0.30 - 0.12 * u, 0.08), 0.02, 0.7)
    inventory = current_assets * np.clip(rng.normal(0.25, 0.08, size=n), 0.0, 0.6)
    receivables = current_assets * np.clip(rng.normal(0.30, 0.08, size=n), 0.0, 0.6)

    revenue = total_assets * np.clip(rng.normal(0.9, 0.25, size=n), 0.2, 2.5)
    net_margin = rng.normal(0.06 - 0.10 * u, 0.05)
    net_income = revenue * net_margin
    ebit = revenue * (net_margin + np.clip(rng.normal(0.04, 0.02, size=n), 0.0, 0.15))
    interest_expense = total_liabilities * np.clip(rng.normal(0.045, 0.012, size=n), 0.01, 0.12)
    long_term_debt = total_liabilities * np.clip(rng.normal(0.55, 0.12, size=n), 0.1, 0.95)
    retained_earnings = equity * np.clip(rng.normal(0.5 - 0.6 * u, 0.25), -1.5, 1.2)

    logits = base_rate_logit + 3.4 * u + 1.2 * (leverage - 0.5) + rng.normal(0, 0.5, size=n)
    label = rng.binomial(1, 1.0 / (1.0 + np.exp(-logits)))

    texts = [_compose_text(rng, float(ui)) for ui in u]

    df = pd.DataFrame(
        {
            "cik": cik,
            "fyear": fyear,
            "total_assets": total_assets,
            "total_liabilities": total_liabilities,
            "current_assets": current_assets,
            "current_liabilities": current_liabilities,
            "cash": cash,
            "inventory": inventory,
            "receivables": receivables,
            "revenue": revenue,
            "net_income": net_income,
            "ebit": ebit,
            "interest_expense": interest_expense,
            "long_term_debt": long_term_debt,
            "equity": equity,
            "retained_earnings": retained_earnings,
            "mda": texts,
            "label": label,
        }
    )
    df["doc_id"] = df["cik"].astype(str) + "_" + df["fyear"].astype(str) + "_" + df.index.astype(str)
    # Plausible Stichtage, damit Event-/Default-Workflows auch auf Synthetik laufen:
    # Bilanzstichtag 31.12., 10-K-Filing ~90 Tage später.
    df["reporting_date"] = pd.to_datetime(df["fyear"].astype(str) + "-12-31")
    df["filing_date"] = df["reporting_date"] + pd.Timedelta(days=90)
    return df


def make_synthetic_events(
    df: pd.DataFrame, *, seed: int = 11, bankruptcy_rate: float = 0.08
) -> pd.DataFrame:
    """Synthetische 8-K-Eventliste passend zu einem Synthetik-Datensatz.

    Erzeugt je Firma mit Wahrscheinlichkeit ``bankruptcy_rate`` eine
    Insolvenzmeldung (Item 1.03) 3–14 Monate nach dem letzten Bilanzstichtag
    sowie verstreute Frühindikator-8-Ks. Ein spätes Dummy-Event schiebt
    ``global_max`` nach hinten, damit die Demo nicht großflächig
    rechtszensiert wird.
    """
    rng = np.random.default_rng(seed)
    frame = df.copy()
    frame["reporting_date"] = pd.to_datetime(frame["reporting_date"])
    frame["filing_date"] = pd.to_datetime(frame["filing_date"])
    # Insolvenzrisiko an den Verschuldungsgrad koppeln (lernbares Signal):
    lev = (
        pd.to_numeric(frame["total_liabilities"], errors="coerce")
        / pd.to_numeric(frame["total_assets"], errors="coerce")
    ).clip(0, 1.5).fillna(0.5)
    frame["_lev"] = lev
    rows: list[tuple[int, pd.Timestamp, str]] = []
    for cik, g in frame.groupby("cik"):
        last_rep = g["reporting_date"].max()
        firm_lev = float(g["_lev"].mean())
        p_bk = bankruptcy_rate * (0.3 + 2.4 * firm_lev)  # ~0.3x bis ~3x der Basisrate
        if rng.random() < p_bk:
            offset = int(rng.integers(90, 420))
            rows.append((int(cik), last_rep + pd.Timedelta(days=offset), "1.03,9.01"))
        for _, r in g.iterrows():
            if rng.random() < 0.25:
                item = str(rng.choice(["2.02", "2.04", "2.06", "4.01", "5.02", "8.01"]))
                back = int(rng.integers(10, 300))
                rows.append((int(cik), r["filing_date"] - pd.Timedelta(days=back), item))
    rows.append((999_999, pd.Timestamp("2035-01-01"), "9.01"))
    events = pd.DataFrame(rows, columns=["cik", "filing_date_8k", "items"])
    events["accession"] = ""
    return events


def make_synthetic_ratings(df: pd.DataFrame, *, seed: int = 13) -> pd.DataFrame:
    """Synthetische Rating-Historie passend zu Firm-Years (FMP-ähnliche Buchstaben).

    Hoher Leverage → schwächerer Notch; je CIK eine vierteljährliche Serie,
    die den Bilanzstichtag PIT-mäßig abdeckt.
    """
    rng = np.random.default_rng(seed)
    frame = df.copy()
    frame["reporting_date"] = pd.to_datetime(frame["reporting_date"])
    lev = (
        pd.to_numeric(frame["total_liabilities"], errors="coerce")
        / pd.to_numeric(frame["total_assets"], errors="coerce")
    ).clip(0, 1.5).fillna(0.5)
    frame["_lev"] = lev
    letters = ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-"]
    rows: list[dict] = []
    for cik, g in frame.groupby("cik"):
        firm_lev = float(g["_lev"].mean())
        base = int(np.clip(round(2 + 6 * firm_lev + rng.normal(0, 0.8)), 0, len(letters) - 1))
        start = g["reporting_date"].min() - pd.DateOffset(months=9)
        end = g["reporting_date"].max() + pd.DateOffset(months=18)
        dates = pd.date_range(start, end, freq="QS")
        notch_i = base
        for dt in dates:
            notch_i = int(np.clip(notch_i + rng.integers(-1, 2), 0, len(letters) - 1))
            rows.append(
                {
                    "cik": int(cik),
                    "ticker": f"T{int(cik)}",
                    "rating_date": dt,
                    "rating": letters[notch_i],
                    "agency": "fmp",
                    "source": "synthetic",
                }
            )
    return pd.DataFrame(rows)
