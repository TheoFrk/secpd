"""Mock-Client: Determinismus, Wertebereiche, heuristische Ordnung."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from secpd.llm.mock import MockLLMClient
from secpd.llm.schema import TextRiskProfile

PLAIN = (
    "Revenue for the year was 500 million dollars. Costs were 300 million dollars. "
    "The company operates three factories. Headcount was 2000 employees. "
    "Net income equaled 40 million dollars for the period under review."
)
HEDGY = (
    "Management believes results may vary substantially and could potentially differ. "
    "We anticipate that certain estimates might possibly require adjustment, although "
    "outcomes are uncertain and assumptions are generally subject to uncertainty. "
    "It appears that liquidity would likely be approximately sufficient, we believe."
)
NEGATIVE = (
    "The company recorded an impairment and disclosed a material weakness. Litigation "
    "over the restatement is pending, a covenant breach occurred, and there is doubt "
    "about going concern. Losses from writedowns increased and a downgrade followed."
)
POSITIVE = (
    "The company achieved record revenues with strong growth and improved margins. "
    "Robust momentum, successful expansion and favorable conditions supported "
    "profitability, and management is confident in continued improvement overall."
)


def test_deterministic() -> None:
    client = MockLLMClient()
    a = client.analyze(HEDGY)
    b = client.analyze(HEDGY)
    assert a == b


def test_bounds() -> None:
    client = MockLLMClient()
    for text in (PLAIN, HEDGY, NEGATIVE, POSITIVE):
        p = client.analyze(text)
        assert 0.0 <= p.vagueness_score <= 1.0
        assert 0.0 <= p.redundancy_score <= 1.0
        assert 0.0 <= p.complexity_score <= 1.0
        assert -1.0 <= p.risk_sentiment <= 1.0


def test_heuristic_ordering() -> None:
    client = MockLLMClient(jitter=0.0)
    assert client.analyze(HEDGY).vagueness_score > client.analyze(PLAIN).vagueness_score
    assert client.analyze(NEGATIVE).risk_sentiment < client.analyze(POSITIVE).risk_sentiment


def test_short_text_fallback() -> None:
    p = MockLLMClient().analyze("Too short.")
    assert p.confidence == 0.0
    assert p.risk_summary.startswith("[fallback]")


def test_feature_projection() -> None:
    p = MockLLMClient().analyze(PLAIN)
    feats = p.to_features()
    assert set(feats) == set(TextRiskProfile.feature_names())
