"""Transformer model wrappers with lazy loading and result caching.

Three models are wrapped behind small classes so the notebook and the
desktop app never touch raw pipelines:

* ``SentimentModel``: robertuito sentiment for Spanish (pysentimiento),
  the primary scorer, returning a label plus NEG, NEU, POS probabilities.
* ``EmotionModel``: robertuito emotion for Spanish (pysentimiento),
  returning the dominant emotion and its probability distribution.
* ``StarModel``: a multilingual review model (nlptown) that grades text
  from 1 to 5 stars, used as an independent second opinion. Its stars are
  mapped to the shared NEG, NEU, POS vocabulary for agreement analysis.

The heavy ML stack (torch, transformers, pysentimiento) is an optional
dependency. Everything degrades with a clear install instruction when it
is absent, and batch results are cached to csv so the analysis can be
reproduced without re running inference.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from .config import DATE_COLUMNS, TEXT_COL

__all__ = [
    "MissingMLDependencies",
    "ml_available",
    "require_ml",
    "SentimentModel",
    "EmotionModel",
    "StarModel",
    "stars_to_sentiment",
    "score_records",
    "save_scored",
    "load_scored",
]

INSTALL_HINT = 'pip install "pqr_sentiment[ml]"'


class MissingMLDependencies(RuntimeError):
    """Raised when the optional transformer stack is not installed."""


def ml_available() -> bool:
    return all(
        importlib.util.find_spec(module) is not None
        for module in ("torch", "transformers", "pysentimiento")
    )


def require_ml() -> None:
    if not ml_available():
        raise MissingMLDependencies(
            "The transformer models are not installed in this environment. "
            f"Install the optional ML stack with: {INSTALL_HINT}"
        )


ProgressCallback = Callable[[int, int], None]


def _iterate_with_progress(texts: list[str], progress: ProgressCallback | None):
    total = len(texts)
    for index, text in enumerate(texts, start=1):
        yield text
        if progress is not None:
            progress(index, total)


class SentimentModel:
    """Spanish sentiment scorer built on pysentimiento robertuito."""

    def __init__(self) -> None:
        require_ml()
        from pysentimiento import create_analyzer

        self._analyzer = create_analyzer(task="sentiment", lang="es")

    def predict(self, text: str) -> tuple[str, dict[str, float]]:
        result = self._analyzer.predict(str(text))
        return result.output, dict(result.probas)

    def score_texts(self, texts: Iterable[str], progress: ProgressCallback | None = None) -> pd.DataFrame:
        rows = []
        for text in _iterate_with_progress([str(t) for t in texts], progress):
            label, probas = self.predict(text)
            rows.append(
                {
                    "sentiment": label,
                    "prob_neg": probas.get("NEG", 0.0),
                    "prob_neu": probas.get("NEU", 0.0),
                    "prob_pos": probas.get("POS", 0.0),
                }
            )
        return pd.DataFrame(rows)


class EmotionModel:
    """Spanish emotion scorer built on pysentimiento robertuito."""

    def __init__(self) -> None:
        require_ml()
        from pysentimiento import create_analyzer

        self._analyzer = create_analyzer(task="emotion", lang="es")

    def predict(self, text: str) -> tuple[str, dict[str, float]]:
        result = self._analyzer.predict(str(text))
        return result.output, dict(result.probas)

    def score_texts(self, texts: Iterable[str], progress: ProgressCallback | None = None) -> pd.DataFrame:
        rows = []
        for text in _iterate_with_progress([str(t) for t in texts], progress):
            label, probas = self.predict(text)
            row = {"emotion": label, "emotion_prob": probas.get(label, 0.0)}
            row.update({f"emo_{name}": value for name, value in probas.items()})
            rows.append(row)
        return pd.DataFrame(rows)


class StarModel:
    """Multilingual 1 to 5 star review grader used as a second opinion."""

    MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"

    def __init__(self) -> None:
        require_ml()
        from transformers import pipeline

        self._pipeline = pipeline("sentiment-analysis", model=self.MODEL_NAME, truncation=True)

    def predict(self, text: str) -> int:
        result = self._pipeline(str(text)[:2000])[0]
        return int(result["label"].split()[0])

    def score_texts(self, texts: Iterable[str], progress: ProgressCallback | None = None) -> pd.DataFrame:
        rows = [
            {"stars": self.predict(text)}
            for text in _iterate_with_progress([str(t) for t in texts], progress)
        ]
        frame = pd.DataFrame(rows)
        frame["stars_sentiment"] = frame["stars"].map(stars_to_sentiment)
        return frame


def stars_to_sentiment(stars: int) -> str:
    """Map a 1 to 5 star grade onto the NEG, NEU, POS vocabulary."""
    if stars not in (1, 2, 3, 4, 5):
        raise ValueError(f"Star grade must be between 1 and 5, got {stars}")
    if stars <= 2:
        return "NEG"
    if stars == 3:
        return "NEU"
    return "POS"


def score_records(
    df: pd.DataFrame,
    text_col: str = TEXT_COL,
    include_emotion: bool = True,
    include_stars: bool = True,
    progress: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Run every requested model over ``df[text_col]`` and append columns."""
    texts = df[text_col].fillna("").tolist()
    pieces = [df.reset_index(drop=True)]

    pieces.append(SentimentModel().score_texts(texts, progress))
    if include_emotion:
        pieces.append(EmotionModel().score_texts(texts, progress))
    if include_stars:
        pieces.append(StarModel().score_texts(texts, progress))

    return pd.concat(pieces, axis=1)


def save_scored(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def load_scored(path: str | Path) -> pd.DataFrame | None:
    path = Path(path)
    if not path.exists():
        return None
    frame = pd.read_csv(path, encoding="utf-8")
    for column in DATE_COLUMNS + ["received_month"]:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame
