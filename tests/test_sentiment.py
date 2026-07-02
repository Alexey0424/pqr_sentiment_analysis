import pandas as pd
import pytest

from pqr_sentiment.sentiment import (
    MissingMLDependencies,
    load_scored,
    require_ml,
    save_scored,
    stars_to_sentiment,
)


class TestStarsMapping:
    @pytest.mark.parametrize(
        "stars,expected",
        [(1, "NEG"), (2, "NEG"), (3, "NEU"), (4, "POS"), (5, "POS")],
    )
    def test_mapping(self, stars, expected):
        assert stars_to_sentiment(stars) == expected

    @pytest.mark.parametrize("stars", [0, 6, -1])
    def test_out_of_range_rejected(self, stars):
        with pytest.raises(ValueError):
            stars_to_sentiment(stars)


class TestScoredCache:
    def test_round_trip(self, tmp_path):
        df = pd.DataFrame(
            {
                "record_id": [1, 2],
                "description": ["texto uno", "texto dos"],
                "sentiment": ["NEG", "NEU"],
                "prob_neg": [0.9, 0.2],
                "prob_neu": [0.08, 0.7],
                "prob_pos": [0.02, 0.1],
                "received_date": pd.to_datetime(["2018-01-05", "2018-02-11"]),
            }
        )
        path = tmp_path / "scored.csv"
        save_scored(df, path)
        loaded = load_scored(path)
        assert loaded is not None
        assert list(loaded.columns) == list(df.columns)
        assert loaded["prob_neg"].iloc[0] == pytest.approx(0.9)
        assert loaded["received_date"].dtype.kind == "M"

    def test_load_missing_returns_none(self, tmp_path):
        assert load_scored(tmp_path / "absent.csv") is None


class TestMissingDependencies:
    def test_require_ml_error_mentions_install_command(self, monkeypatch):
        import pqr_sentiment.sentiment as sentiment_module

        monkeypatch.setattr(sentiment_module, "ml_available", lambda: False)
        with pytest.raises(MissingMLDependencies, match=r"pip install"):
            require_ml()
