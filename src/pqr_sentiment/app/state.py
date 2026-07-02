"""Application state and background work.

Views read from this object and subscribe to change notifications. All
model inference runs in worker threads; callbacks are marshalled back to
the Tk main loop by the views through ``widget.after``.
"""

from __future__ import annotations

import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from .. import config
from ..data import CleaningReport, clean_records, load_records
from ..sentiment import (
    EmotionModel,
    MissingMLDependencies,
    SentimentModel,
    ml_available,
    require_ml,
)

Listener = Callable[[str], None]


@dataclass
class AppState:
    source_name: str | None = None
    records: pd.DataFrame | None = None
    report: CleaningReport | None = None
    scored: pd.DataFrame | None = None
    scoring_active: bool = False
    _sentiment_model: SentimentModel | None = None
    _emotion_model: EmotionModel | None = None
    _model_lock: threading.Lock = field(default_factory=threading.Lock)
    _listeners: list[Listener] = field(default_factory=list)

    # ------------------------------------------------------------------ events
    def subscribe(self, listener: Listener) -> None:
        self._listeners.append(listener)

    def notify(self, event: str) -> None:
        for listener in list(self._listeners):
            listener(event)

    # ------------------------------------------------------------------ data
    def load_file(self, path: str | Path) -> None:
        raw = load_records(path)
        self.records, self.report = clean_records(raw)
        self.scored = None
        self.source_name = Path(path).name
        self.notify("data")

    def load_demo(self) -> bool:
        path = config.find_data_file(config.PUBLIC_DATASET_NAME)
        if path is None:
            return False
        self.load_file(path)
        scored_path = config.find_data_file(config.SCORED_DATASET_NAME)
        if scored_path is not None:
            from ..sentiment import load_scored

            cached = load_scored(scored_path)
            if cached is not None and len(cached) == len(self.records):
                self.scored = cached
                self.notify("scored")
        return True

    # ------------------------------------------------------------------ models
    def models_available(self) -> bool:
        return ml_available()

    def ensure_models(self) -> tuple[SentimentModel, EmotionModel]:
        with self._model_lock:
            if self._sentiment_model is None:
                require_ml()
                self._sentiment_model = SentimentModel()
            if self._emotion_model is None:
                self._emotion_model = EmotionModel()
        return self._sentiment_model, self._emotion_model

    def warm_models_async(self, done: Callable[[Exception | None], None]) -> None:
        def work() -> None:
            try:
                self.ensure_models()
                done(None)
            except Exception as error:  # noqa: BLE001 - surfaced to the UI
                done(error)

        threading.Thread(target=work, daemon=True).start()

    # ------------------------------------------------------------------ scoring
    def analyze_text(self, text: str) -> dict:
        sentiment_model, emotion_model = self.ensure_models()
        label, probas = sentiment_model.predict(text)
        emotion, emotion_probas = emotion_model.predict(text)
        return {
            "sentiment": label,
            "probas": probas,
            "emotion": emotion,
            "emotion_probas": emotion_probas,
        }

    def score_dataset_async(
        self,
        progress: Callable[[int, int, str], None],
        done: Callable[[Exception | None], None],
    ) -> None:
        if self.records is None or self.scoring_active:
            return
        self.scoring_active = True

        def work() -> None:
            try:
                sentiment_model, emotion_model = self.ensure_models()
                records = self.records.reset_index(drop=True)
                texts = records[config.TEXT_COL].fillna("").tolist()

                sentiment = sentiment_model.score_texts(
                    texts, lambda i, n: progress(i, n, "sentiment")
                )
                emotion = emotion_model.score_texts(
                    texts, lambda i, n: progress(i, n, "emotion")
                )
                self.scored = pd.concat([records, sentiment, emotion], axis=1)
                self.scoring_active = False
                done(None)
            except Exception as error:  # noqa: BLE001 - surfaced to the UI
                traceback.print_exc()
                self.scoring_active = False
                done(error)

        threading.Thread(target=work, daemon=True).start()

    # ------------------------------------------------------------------ export
    def export(self, directory: str | Path) -> list[Path]:
        from .. import plots
        from ..text_mining import top_terms_by_class

        if self.scored is None:
            raise RuntimeError("Nothing to export yet: score a dataset first.")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []

        csv_path = directory / "pqr_scored.csv"
        self.scored.to_csv(csv_path, index=False, encoding="utf-8")
        written.append(csv_path)

        plots.apply_theme(dark=True)
        figures = {
            "sentiment_by_type": lambda: plots.sentiment_share_by_category(
                self.scored, "pqr_type", "Sentiment by record type", display=config.VALUE_DISPLAY
            ),
            "sentiment_by_municipality": lambda: plots.sentiment_share_by_category(
                self.scored, "municipality", "Sentiment by municipality"
            ),
            "monthly_activity": lambda: plots.monthly_activity(
                self.scored, "Monthly volume and negative share"
            ),
            "emotion_distribution": lambda: plots.emotion_distribution(
                self.scored, "Dominant emotions"
            ),
            "distinctive_terms": lambda: plots.terms_panel(
                top_terms_by_class(self.scored, config.TEXT_COL, "sentiment", top_n=10),
                "Distinctive terms per sentiment",
            ),
        }
        for name, build in figures.items():
            try:
                figure = build()
                written.append(plots.save_figure(figure, name, directory))
            except Exception:  # noqa: BLE001 - export the rest anyway
                traceback.print_exc()
        return written


MISSING_ML_MESSAGE = (
    "The transformer models are not installed in this environment.\n"
    'Install them with:  pip install "pqr_sentiment[ml]"\n'
    "then restart the studio."
)

__all__ = ["AppState", "MissingMLDependencies", "MISSING_ML_MESSAGE"]
