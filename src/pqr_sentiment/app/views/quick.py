"""Quick Analysis: paste any Spanish text, read its sentiment and emotion."""

from __future__ import annotations

import threading

import customtkinter as ctk

from .. import theming
from ..state import MISSING_ML_MESSAGE, AppState
from ..widgets import ProbabilityRow, Section, body_font, heading_font

PLACEHOLDER = (
    "Paste a PQR narrative here, for example:\n\n"
    "Durante los trabajos de reposición de alcantarillado la maquinaria "
    "afectó la fachada de mi vivienda y nadie ha respondido por el arreglo."
)

EMOTION_LABELS = {
    "anger": "Anger", "sadness": "Sadness", "disgust": "Disgust", "fear": "Fear",
    "surprise": "Surprise", "joy": "Joy", "others": "No dominant emotion",
}


class QuickView(ctk.CTkFrame):
    def __init__(self, master, state: AppState, set_status) -> None:
        super().__init__(master, fg_color="transparent")
        self.state = state
        self.set_status = set_status

        ctk.CTkLabel(
            self, text="Quick Analysis", font=heading_font(22),
            text_color=theming.INK, anchor="w",
        ).pack(fill="x", pady=(0, 2))
        ctk.CTkLabel(
            self,
            text="Score any Spanish narrative in place: the sentiment triple and the emotion probe, live.",
            font=body_font(13), text_color=theming.MUTED, anchor="w",
        ).pack(fill="x", pady=(0, 12))

        input_card = Section(self, "Narrative")
        input_card.pack(fill="x")
        self.textbox = ctk.CTkTextbox(
            input_card, height=140, corner_radius=8,
            fg_color=theming.SURFACE, text_color=theming.INK,
            border_width=1, border_color=theming.CARD_BORDER,
            font=body_font(13), wrap="word",
        )
        self.textbox.pack(fill="x", padx=16, pady=(0, 10))
        self.textbox.insert("1.0", PLACEHOLDER)

        actions = ctk.CTkFrame(input_card, fg_color="transparent")
        actions.pack(fill="x", padx=16, pady=(0, 14))
        self.analyze_button = ctk.CTkButton(
            actions, text="Analyze", width=140, height=34, corner_radius=8,
            fg_color=theming.ACCENT, hover_color=theming.ACCENT_HOVER,
            font=body_font(13, "bold"), command=self.analyze,
        )
        self.analyze_button.pack(side="left")
        self.clear_button = ctk.CTkButton(
            actions, text="Clear", width=90, height=34, corner_radius=8,
            fg_color=theming.CARD_BORDER, hover_color="#3a3a37",
            font=body_font(13), command=self.clear,
        )
        self.clear_button.pack(side="left", padx=8)

        self.sentiment_card = Section(self, "Sentiment")
        self.sentiment_card.pack(fill="x", pady=(12, 0))
        self.sentiment_headline = ctk.CTkLabel(
            self.sentiment_card, text="Run an analysis to see results.",
            font=heading_font(16), text_color=theming.MUTED, anchor="w",
        )
        self.sentiment_headline.pack(fill="x", padx=16, pady=(0, 6))
        self.sentiment_rows: dict[str, ProbabilityRow] = {}
        for label in ("NEG", "NEU", "POS"):
            row = ProbabilityRow(
                self.sentiment_card, theming.SENTIMENT_LABELS[label], theming.SENTIMENT[label]
            )
            row.pack(fill="x", padx=16, pady=3)
            self.sentiment_rows[label] = row
        ctk.CTkLabel(self.sentiment_card, text="", height=4).pack()

        self.emotion_card = Section(self, "Emotion")
        self.emotion_card.pack(fill="x", pady=(12, 0))
        self.emotion_headline = ctk.CTkLabel(
            self.emotion_card, text="", font=body_font(14),
            text_color=theming.INK_SECONDARY, anchor="w",
        )
        self.emotion_headline.pack(fill="x", padx=16, pady=(0, 10))

        self.error_label = ctk.CTkLabel(
            self, text="", font=body_font(12), text_color="#e66767",
            anchor="w", justify="left",
        )
        self.error_label.pack(fill="x", pady=(10, 0))

    # ------------------------------------------------------------------
    def clear(self) -> None:
        self.textbox.delete("1.0", "end")
        for row in self.sentiment_rows.values():
            row.reset()
        self.sentiment_headline.configure(text="Run an analysis to see results.", text_color=theming.MUTED)
        self.emotion_headline.configure(text="")
        self.error_label.configure(text="")

    def analyze(self) -> None:
        text = self.textbox.get("1.0", "end").strip()
        if not text:
            return
        if not self.state.models_available():
            self.error_label.configure(text=MISSING_ML_MESSAGE)
            return
        self.analyze_button.configure(state="disabled", text="Analyzing...")
        self.set_status("Scoring narrative, first run loads the models...")

        def work() -> None:
            try:
                result = self.state.analyze_text(text)
                self.after(0, lambda: self._render(result))
            except Exception as error:  # noqa: BLE001 - shown in the UI
                self.after(0, lambda: self._fail(error))

        threading.Thread(target=work, daemon=True).start()

    def _render(self, result: dict) -> None:
        label = result["sentiment"]
        self.sentiment_headline.configure(
            text=f"{theming.SENTIMENT_LABELS[label]}  "
                 f"({result['probas'][label] * 100:.0f}% confidence)",
            text_color=theming.SENTIMENT[label],
        )
        for name, row in self.sentiment_rows.items():
            row.set(result["probas"].get(name, 0.0))
        emotion = result["emotion"]
        share = result["emotion_probas"].get(emotion, 0.0)
        self.emotion_headline.configure(
            text=f"{EMOTION_LABELS.get(emotion, emotion)}  ({share * 100:.0f}%)"
        )
        self.error_label.configure(text="")
        self.analyze_button.configure(state="normal", text="Analyze")
        self.set_status("Ready.")

    def _fail(self, error: Exception) -> None:
        self.error_label.configure(text=str(error))
        self.analyze_button.configure(state="normal", text="Analyze")
        self.set_status("Scoring failed.")
