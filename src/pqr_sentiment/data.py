"""Loading, validation and cleaning of PQR records.

The source system exports carry every classic operational data defect:
category values with stray whitespace, free case labels, neighborhood
names typed twice with different capitalization, Excel serial dates mixed
with ISO and day first strings, and the occasional truncated date that no
parser can recover. ``clean_records`` fixes what can be fixed, drops what
cannot, and accounts for every decision in a :class:`CleaningReport` so the
analysis never hides its preprocessing.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .config import (
    DATE_COLUMNS,
    ESSENTIAL_COLUMNS,
    RAW_COLUMNS,
    VALUE_MAPS,
)

__all__ = ["DataValidationError", "CleaningReport", "load_records", "parse_flexible_dates", "clean_records"]


class DataValidationError(ValueError):
    """Raised when an input file does not match the expected schema."""


def load_records(path: str | Path) -> pd.DataFrame:
    """Read a records file (csv or Excel) and validate its schema."""
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, encoding="utf-8")
    missing = [column for column in RAW_COLUMNS if column not in df.columns]
    if missing:
        raise DataValidationError(f"Input file is missing expected columns: {missing}")
    return df[RAW_COLUMNS].copy()


# Excel serials for the plausible operational window (1954 to 2064). Bare
# numbers outside this range, like a stray year, are not treated as serials.
_SERIAL_MIN, _SERIAL_MAX = 20_000, 60_000
_EXCEL_EPOCH = "1899-12-30"


def parse_flexible_dates(series: pd.Series) -> pd.Series:
    """Parse a column that mixes Excel serials, ISO strings and day first strings."""
    values = series.copy()
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")

    numeric = pd.to_numeric(values, errors="coerce")
    serial_mask = numeric.between(_SERIAL_MIN, _SERIAL_MAX)
    if serial_mask.any():
        parsed.loc[serial_mask] = pd.to_datetime(
            numeric[serial_mask], unit="D", origin=_EXCEL_EPOCH, errors="coerce"
        )

    text_mask = ~serial_mask & values.notna()
    if text_mask.any():
        text = values[text_mask].astype(str).str.strip(" '\"\t")
        parsed.loc[text_mask] = _parse_text_stage(text, {"format": "ISO8601"})
        remaining = text_mask & parsed.isna()
        if remaining.any():
            text = values[remaining].astype(str).str.strip(" '\"\t")
            parsed.loc[remaining] = _parse_text_stage(text, {"format": "%d/%m/%Y"})
        remaining = text_mask & parsed.isna()
        if remaining.any():
            text = values[remaining].astype(str).str.strip(" '\"\t")
            parsed.loc[remaining] = _parse_text_stage(text, {"dayfirst": True})

    return parsed.dt.normalize()


# Values outside this window are source system garbage (a year truncated to
# "201" parses as the third century on lenient parsers).
_PLAUSIBLE_MIN = pd.Timestamp("1950-01-01")
_PLAUSIBLE_MAX = pd.Timestamp("2100-12-31")


def _parse_text_stage(text: pd.Series, kwargs: dict) -> pd.Series:
    """Parse one text stage without ever raising.

    Depending on the pandas version, a value like ``06/05/201`` either
    coerces to NaT, raises OutOfBoundsDatetime, or parses successfully into
    a lower resolution timestamp that later explodes when assigned into a
    nanosecond column. Parsing per stage and discarding anything outside
    the plausible operational window neutralizes all three behaviors.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            result = pd.to_datetime(text, errors="coerce", **kwargs)
        except Exception:
            parsed_values = []
            for value in text:
                try:
                    parsed_values.append(pd.to_datetime(value, errors="coerce", **kwargs))
                except Exception:
                    parsed_values.append(pd.NaT)
            result = pd.Series(parsed_values, index=text.index)
    result = pd.Series(result, index=text.index)
    out_of_window = result.notna() & ((result < _PLAUSIBLE_MIN) | (result > _PLAUSIBLE_MAX))
    result[out_of_window] = pd.NaT
    return result.astype("datetime64[ns]")


@dataclass
class CleaningReport:
    """Accounting of every transformation applied by :func:`clean_records`."""

    rows_in: int
    rows_out: int
    fixes: dict[str, int] = field(default_factory=dict)
    dropped: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_frame(self) -> pd.DataFrame:
        rows = [
            {"metric": "rows_in", "value": self.rows_in},
            {"metric": "rows_out", "value": self.rows_out},
            {"metric": "rows_dropped", "value": self.rows_in - self.rows_out},
        ]
        rows.extend({"metric": name, "value": count} for name, count in sorted(self.fixes.items()))
        return pd.DataFrame(rows)


def _strip_whitespace(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    changed = 0
    for column in df.columns:
        if df[column].dtype == object:
            stripped = df[column].map(lambda v: " ".join(v.split()) if isinstance(v, str) else v)
            changed += int((stripped != df[column]).fillna(False).sum())
            df[column] = stripped
    return df, changed


def _map_category_values(df: pd.DataFrame) -> pd.DataFrame:
    for column, mapping in VALUE_MAPS.items():
        normalized = df[column].map(lambda v: mapping.get(v.casefold(), v) if isinstance(v, str) else v)
        df[column] = normalized
    return df


def _merge_neighborhood_variants(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Merge spellings that differ only by case to the most frequent variant."""
    names = df["neighborhood"].dropna().astype(str)
    counts = names.value_counts()
    canonical: dict[str, str] = {}
    for key, group in names.groupby(names.str.casefold()):
        variants = counts.loc[counts.index.isin(group.unique())]
        winner = sorted(variants.index, key=lambda name: (-variants[name], name))[0]
        canonical[key] = winner
    merged = df["neighborhood"].map(
        lambda v: canonical.get(v.casefold(), v) if isinstance(v, str) else v
    )
    changed = int((merged != df["neighborhood"]).fillna(False).sum())
    df["neighborhood"] = merged
    return df, changed


def clean_records(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Normalize categories, repair dates, drop unusable rows, derive fields."""
    frame = df.copy()
    fixes: dict[str, int] = {}

    frame, trimmed = _strip_whitespace(frame)
    fixes["whitespace_trimmed_cells"] = trimmed

    frame = _map_category_values(frame)

    frame, merged = _merge_neighborhood_variants(frame)
    fixes["neighborhood_variants_merged"] = merged

    unparseable = 0
    for column in DATE_COLUMNS:
        parsed = parse_flexible_dates(frame[column])
        unparseable += int((frame[column].notna() & parsed.isna()).sum())
        frame[column] = parsed
    fixes["unparseable_date_values"] = unparseable

    essentials = frame[ESSENTIAL_COLUMNS].replace("", pd.NA)
    missing_mask = essentials.isna()
    drop_mask = missing_mask.any(axis=1)
    dropped = frame.loc[drop_mask].copy()
    if len(dropped):
        dropped["drop_reason"] = missing_mask.loc[drop_mask].apply(
            lambda row: "missing " + ", ".join(column for column in ESSENTIAL_COLUMNS if row[column]),
            axis=1,
        )
    fixes["rows_dropped_missing_essentials"] = int(drop_mask.sum())
    frame = frame.loc[~drop_mask].reset_index(drop=True)

    frame["resolution_days"] = (frame["closed_date"] - frame["received_date"]).dt.days
    sla = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    known = frame["closed_date"].notna() & frame["due_date"].notna()
    sla.loc[known] = frame.loc[known, "closed_date"] <= frame.loc[known, "due_date"]
    frame["sla_met"] = sla
    frame["received_month"] = frame["received_date"].dt.to_period("M").dt.to_timestamp()

    report = CleaningReport(rows_in=len(df), rows_out=len(frame), fixes=fixes, dropped=dropped)
    return frame, report
