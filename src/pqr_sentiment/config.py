"""Central configuration: schema names, category mappings and shared constants.

The public dataset ships with English snake_case column names while the
categorical values keep the exact raw Spanish spellings found in the source
system, including their defects. The cleaning pipeline normalizes those
values and maps them to the English codes defined here, so every downstream
consumer (notebook, app, tests) speaks a single vocabulary.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Public dataset schema
# ---------------------------------------------------------------------------

RECORD_ID = "record_id"
TEXT_COL = "description"

DATE_COLUMNS = ["received_date", "due_date", "attended_date", "closed_date"]

RAW_COLUMNS = [
    "record_id",
    "pqr_type",
    "received_date",
    "due_date",
    "attended_date",
    "closed_date",
    "entry_route",
    "watershed",
    "municipality",
    "neighborhood",
    "description",
    "attributable",
    "status",
]

# Rows missing any of these fields cannot be analyzed and are dropped with a
# reason recorded in the cleaning report.
ESSENTIAL_COLUMNS = [
    "pqr_type",
    "received_date",
    "description",
    "municipality",
    "watershed",
    "neighborhood",
]

CATEGORY_COLUMNS = ["pqr_type", "entry_route", "watershed", "municipality", "neighborhood", "attributable", "status"]

# ---------------------------------------------------------------------------
# Category value normalization (raw Spanish -> English codes)
# Keys are compared after strip and casefold.
# ---------------------------------------------------------------------------

PQR_TYPE_MAP = {
    "petición": "petition",
    "peticion": "petition",
    "queja": "complaint",
    "reclamo": "claim",
}

ENTRY_ROUTE_MAP = {
    "contratista": "contractor",
    "interventoría": "supervision",
    "interventoria": "supervision",
    "empresa de servicios": "utility_company",
}

ATTRIBUTABLE_MAP = {
    "imputable": "attributable",
    "no imputable": "not_attributable",
    "compartida": "shared",
}

STATUS_MAP = {
    "cerrada": "closed",
    "abierta": "open",
}

VALUE_MAPS = {
    "pqr_type": PQR_TYPE_MAP,
    "entry_route": ENTRY_ROUTE_MAP,
    "attributable": ATTRIBUTABLE_MAP,
    "status": STATUS_MAP,
}

# ---------------------------------------------------------------------------
# Sentiment and emotion vocabulary
# ---------------------------------------------------------------------------

SENTIMENT_ORDER = ["NEG", "NEU", "POS"]
SENTIMENT_DISPLAY = {"NEG": "Negative", "NEU": "Neutral", "POS": "Positive"}

EMOTION_ORDER = ["anger", "sadness", "disgust", "fear", "surprise", "joy", "others"]
EMOTION_DISPLAY = {
    "anger": "Anger",
    "sadness": "Sadness",
    "disgust": "Disgust",
    "fear": "Fear",
    "surprise": "Surprise",
    "joy": "Joy",
    "others": "No dominant emotion",
}

# Display titles for chart axes and headers.
COLUMN_DISPLAY = {
    "pqr_type": "PQR type",
    "entry_route": "Entry route",
    "watershed": "Watershed",
    "municipality": "Municipality",
    "neighborhood": "Neighborhood",
    "attributable": "Attributability",
    "status": "Status",
    "sentiment": "Sentiment",
    "emotion": "Dominant emotion",
    "resolution_days": "Days to close",
}

VALUE_DISPLAY = {
    "petition": "Petition",
    "complaint": "Complaint",
    "claim": "Claim",
    "contractor": "Contractor",
    "supervision": "Supervision",
    "utility_company": "Utility company",
    "attributable": "Attributable",
    "not_attributable": "Not attributable",
    "shared": "Shared",
    "closed": "Closed",
    "open": "Open",
}

# ---------------------------------------------------------------------------
# Data file discovery
# ---------------------------------------------------------------------------

PUBLIC_DATASET_NAME = "pqr_records.csv"
SCORED_DATASET_NAME = "pqr_scored.csv"


def find_data_file(name: str) -> Path | None:
    """Locate a bundled data file, searching the working tree first."""
    candidates = [
        Path.cwd() / "data" / name,
        Path.cwd() / name,
        Path(__file__).resolve().parents[2] / "data" / name,
    ]
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        candidates.append(Path(bundle_root) / "data" / name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None
