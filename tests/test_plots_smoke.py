import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from pqr_sentiment import plots


@pytest.fixture(scope="module")
def scored() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n = 120
    labels = rng.choice(["NEG", "NEU", "POS"], size=n, p=[0.6, 0.3, 0.1])
    months = pd.to_datetime(rng.choice(pd.date_range("2018-01-01", periods=18, freq="MS"), size=n))
    frame = pd.DataFrame(
        {
            "pqr_type": rng.choice(["complaint", "claim", "petition"], size=n),
            "municipality": rng.choice(["Alpha", "Beta", "Gamma"], size=n),
            "sentiment": labels,
            "prob_neg": rng.uniform(0, 1, size=n),
            "prob_neu": rng.uniform(0, 1, size=n),
            "prob_pos": rng.uniform(0, 1, size=n),
            "emotion": rng.choice(["anger", "sadness", "others", "joy"], size=n),
            "stars_sentiment": rng.choice(["NEG", "NEU", "POS"], size=n),
            "resolution_days": rng.integers(0, 40, size=n).astype(float),
            "received_month": months,
        }
    )
    return frame


@pytest.fixture(scope="module", autouse=True)
def theme():
    plots.apply_theme(dark=False)


def test_volume_by_category(scored):
    fig = plots.volume_by_category(scored, "municipality", "Records by municipality")
    assert fig.axes


def test_sentiment_share_by_category(scored):
    fig = plots.sentiment_share_by_category(scored, "pqr_type", "Sentiment by type")
    legend = fig.axes[0].get_legend()
    assert legend is not None
    assert [t.get_text() for t in legend.get_texts()] == ["Negative", "Neutral", "Positive"]


def test_monthly_activity_has_two_panels_one_clock(scored):
    fig = plots.monthly_activity(scored, "Monthly activity")
    assert len(fig.axes) == 2


def test_probability_distributions(scored):
    fig = plots.probability_distributions(scored, "Model confidence")
    assert len(fig.axes) == 3


def test_emotion_charts(scored):
    assert plots.emotion_distribution(scored, "Dominant emotions").axes
    assert plots.emotion_by_type_heatmap(scored, "Emotion by type").axes


def test_resolution_by_sentiment(scored):
    fig = plots.resolution_by_sentiment(scored, "Resolution time")
    assert fig.axes


def test_agreement_heatmap(scored):
    fig = plots.agreement_heatmap(scored, "Model agreement")
    assert fig.axes


def test_terms_panel(scored):
    terms = pd.DataFrame(
        {
            "class": ["NEG"] * 3 + ["NEU"] * 3 + ["POS"] * 3,
            "term": [f"term {i}" for i in range(9)],
            "score": np.linspace(0.9, 0.1, 9),
        }
    )
    fig = plots.terms_panel(terms, "Distinctive terms")
    assert len(fig.axes) == 3


def test_dark_theme_switches_colors(scored):
    plots.apply_theme(dark=True)
    try:
        assert plots.current_theme() is plots.THEMES["dark"]
        fig = plots.volume_by_category(scored, "municipality", "Dark mode")
        assert fig.get_facecolor()[:3] != (1.0, 1.0, 1.0)
    finally:
        plots.apply_theme(dark=False)


def test_save_figure(tmp_path, scored):
    fig = plots.volume_by_category(scored, "municipality", "Save test")
    path = plots.save_figure(fig, "smoke", directory=tmp_path)
    assert path.exists() and path.stat().st_size > 0
