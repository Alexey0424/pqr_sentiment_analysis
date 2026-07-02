import numpy as np
import pandas as pd
import pytest

from pqr_sentiment.stats import association, kruskal_by_group, share_table


def frame_from_counts(counts: dict[tuple[str, str], int]) -> pd.DataFrame:
    rows = []
    for (a, b), n in counts.items():
        rows.extend({"cat": a, "label": b} for _ in range(n))
    return pd.DataFrame(rows)


class TestAssociation:
    def test_perfect_association_has_high_cramers_v(self):
        df = frame_from_counts({("north", "NEG"): 40, ("south", "POS"): 40})
        result = association(df, "cat", "label")
        assert result.cramers_v > 0.9
        assert result.p_value < 0.001
        assert result.dof == 1

    def test_independent_variables_have_low_cramers_v(self):
        df = frame_from_counts(
            {("north", "NEG"): 30, ("north", "POS"): 30, ("south", "NEG"): 30, ("south", "POS"): 30}
        )
        result = association(df, "cat", "label")
        assert result.cramers_v < 0.05
        assert result.p_value > 0.5

    def test_table_holds_observed_counts(self):
        df = frame_from_counts({("north", "NEG"): 3, ("south", "POS"): 5})
        result = association(df, "cat", "label")
        assert result.table.loc["north", "NEG"] == 3
        assert result.table.loc["south", "POS"] == 5


class TestKruskal:
    def test_shifted_groups_detected(self):
        rng = np.random.default_rng(7)
        df = pd.DataFrame(
            {
                "value": np.concatenate([rng.normal(0, 1, 60), rng.normal(3, 1, 60)]),
                "group": ["a"] * 60 + ["b"] * 60,
            }
        )
        result = kruskal_by_group(df, "value", "group")
        assert result.p_value < 0.001
        assert result.medians["b"] > result.medians["a"]
        assert result.n["a"] == 60

    def test_null_values_ignored(self):
        df = pd.DataFrame({"value": [1.0, 2.0, None, 4.0], "group": ["a", "a", "b", "b"]})
        result = kruskal_by_group(df, "value", "group")
        assert result.n["b"] == 1


class TestShareTable:
    def test_rows_sum_to_one(self):
        df = frame_from_counts({("north", "NEG"): 6, ("north", "POS"): 2, ("south", "NEG"): 1, ("south", "POS"): 3})
        table = share_table(df, "cat", "label")
        assert table.loc["north"].sum() == pytest.approx(1.0)
        assert table.loc["north", "NEG"] == pytest.approx(0.75)

    def test_explicit_order_and_missing_labels(self):
        df = frame_from_counts({("north", "NEG"): 4})
        table = share_table(df, "cat", "label", order=["NEG", "NEU", "POS"])
        assert list(table.columns) == ["NEG", "NEU", "POS"]
        assert table.loc["north", "POS"] == 0.0
