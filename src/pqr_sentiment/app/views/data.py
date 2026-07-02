"""Data view: load a file or the demo dataset, review cleaning, run scoring."""

from __future__ import annotations

import tkinter.ttk as ttk
from tkinter import filedialog

import customtkinter as ctk
import pandas as pd

from .. import theming
from ..state import MISSING_ML_MESSAGE, AppState
from ..widgets import Section, body_font, heading_font


class DataView(ctk.CTkFrame):
    def __init__(self, master, state: AppState, set_status) -> None:
        super().__init__(master, fg_color="transparent")
        self.state = state
        self.set_status = set_status

        ctk.CTkLabel(
            self, text="Data", font=heading_font(22), text_color=theming.INK, anchor="w",
        ).pack(fill="x", pady=(0, 2))
        ctk.CTkLabel(
            self,
            text="Load a PQR export (csv or Excel with the documented schema), or start from the bundled demo dataset.",
            font=body_font(13), text_color=theming.MUTED, anchor="w",
        ).pack(fill="x", pady=(0, 12))

        actions_card = Section(self, "Source")
        actions_card.pack(fill="x")
        actions = ctk.CTkFrame(actions_card, fg_color="transparent")
        actions.pack(fill="x", padx=16, pady=(0, 8))
        ctk.CTkButton(
            actions, text="Open file...", width=140, height=34, corner_radius=8,
            fg_color=theming.ACCENT, hover_color=theming.ACCENT_HOVER,
            font=body_font(13, "bold"), command=self.open_file,
        ).pack(side="left")
        ctk.CTkButton(
            actions, text="Load demo dataset", width=170, height=34, corner_radius=8,
            fg_color=theming.CARD_BORDER, hover_color="#3a3a37",
            font=body_font(13), command=self.load_demo,
        ).pack(side="left", padx=8)
        self.source_label = ctk.CTkLabel(
            actions_card, text="No dataset loaded.", font=body_font(12),
            text_color=theming.MUTED, anchor="w",
        )
        self.source_label.pack(fill="x", padx=16, pady=(0, 14))

        self.report_card = Section(
            self, "Cleaning report",
            "Whitespace, category variants, date repair and dropped rows, all accounted for.",
        )
        self.report_card.pack(fill="x", pady=(12, 0))
        self.report_table = self._make_table(self.report_card, ["metric", "value"], height=7)

        self.scoring_card = Section(self, "Batch scoring")
        self.scoring_card.pack(fill="x", pady=(12, 0))
        scoring_row = ctk.CTkFrame(self.scoring_card, fg_color="transparent")
        scoring_row.pack(fill="x", padx=16, pady=(0, 6))
        self.score_button = ctk.CTkButton(
            scoring_row, text="Run sentiment and emotion scoring", width=260, height=34,
            corner_radius=8, fg_color=theming.ACCENT, hover_color=theming.ACCENT_HOVER,
            font=body_font(13, "bold"), command=self.run_scoring, state="disabled",
        )
        self.score_button.pack(side="left")
        self.progress = ctk.CTkProgressBar(
            scoring_row, height=12, corner_radius=6,
            progress_color=theming.ACCENT, fg_color=theming.CARD_BORDER,
        )
        self.progress.pack(side="left", fill="x", expand=True, padx=(12, 0))
        self.progress.set(0)
        self.scoring_label = ctk.CTkLabel(
            self.scoring_card, text="Load a dataset first.", font=body_font(12),
            text_color=theming.MUTED, anchor="w", justify="left",
        )
        self.scoring_label.pack(fill="x", padx=16, pady=(0, 14))

        self.preview_card = Section(self, "Preview")
        self.preview_card.pack(fill="both", expand=True, pady=(12, 0))
        self.preview_table = self._make_table(
            self.preview_card,
            ["record_id", "pqr_type", "received", "municipality", "description"],
            height=8,
        )

        self.state.subscribe(self.on_state_event)

    # ------------------------------------------------------------------
    def _make_table(self, parent, columns: list[str], height: int) -> ttk.Treeview:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Studio.Treeview",
            background=theming.SURFACE, fieldbackground=theming.SURFACE,
            foreground=theming.INK_SECONDARY, borderwidth=0, rowheight=26,
            font=(theming.FONT_FAMILY, 10),
        )
        style.configure(
            "Studio.Treeview.Heading",
            background=theming.CARD, foreground=theming.MUTED,
            borderwidth=0, font=(theming.FONT_FAMILY, 10, "bold"),
        )
        style.map("Studio.Treeview", background=[("selected", theming.CARD_BORDER)])
        table = ttk.Treeview(
            parent, columns=columns, show="headings", height=height, style="Studio.Treeview",
        )
        for column in columns:
            table.heading(column, text=column)
            width = 340 if column == "description" else (110 if column != "metric" else 240)
            table.column(column, width=width, anchor="w")
        table.pack(fill="both", expand=True, padx=16, pady=(0, 14))
        return table

    def open_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select a PQR export",
            filetypes=[("Data files", "*.csv *.xlsx *.xlsm *.xls"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.state.load_file(path)
            self.set_status(f"Loaded {path}")
        except Exception as error:  # noqa: BLE001 - shown in the UI
            self.source_label.configure(text=str(error), text_color="#e66767")

    def load_demo(self) -> None:
        if not self.state.load_demo():
            self.source_label.configure(
                text="Demo dataset not found next to the application.", text_color="#e66767",
            )
            return
        self.set_status("Demo dataset loaded.")

    # ------------------------------------------------------------------
    def on_state_event(self, event: str) -> None:
        if event == "data":
            self.render_data()
        if event == "scored":
            self.scoring_label.configure(
                text="Scores ready: open the Dashboard view.", text_color=theming.INK_SECONDARY,
            )
            self.progress.set(1.0)

    def render_data(self) -> None:
        records, report = self.state.records, self.state.report
        self.source_label.configure(
            text=f"{self.state.source_name}: {len(records)} analyzable records "
                 f"({report.rows_in} in file)",
            text_color=theming.INK_SECONDARY,
        )
        self.report_table.delete(*self.report_table.get_children())
        for _, row in report.to_frame().iterrows():
            self.report_table.insert("", "end", values=(row["metric"], row["value"]))
        self.preview_table.delete(*self.preview_table.get_children())
        for _, row in records.head(40).iterrows():
            received = row["received_date"]
            self.preview_table.insert(
                "", "end",
                values=(
                    row["record_id"], row["pqr_type"],
                    "" if pd.isna(received) else str(received.date()),
                    row["municipality"], str(row["description"])[:160],
                ),
            )
        self.score_button.configure(state="normal")
        if self.state.scored is not None:
            self.scoring_label.configure(
                text="Cached scores were found for this dataset: open the Dashboard view, "
                     "or rerun scoring to refresh them.",
                text_color=theming.INK_SECONDARY,
            )
            self.progress.set(1.0)
        else:
            self.scoring_label.configure(
                text="Two transformer models run locally over every narrative. "
                     "The first run downloads them once.",
                text_color=theming.MUTED,
            )
            self.progress.set(0)

    # ------------------------------------------------------------------
    def run_scoring(self) -> None:
        if not self.state.models_available():
            self.scoring_label.configure(text=MISSING_ML_MESSAGE, text_color="#e66767")
            return
        self.score_button.configure(state="disabled")
        self.progress.set(0)
        self.scoring_label.configure(text="Loading models...", text_color=theming.INK_SECONDARY)
        self.set_status("Scoring dataset...")

        def on_progress(done: int, total: int, stage: str) -> None:
            base = 0.0 if stage == "sentiment" else 0.5
            fraction = base + (done / total) * 0.5
            self.after(0, lambda: self._progress_update(fraction, stage, done, total))

        def on_done(error: Exception | None) -> None:
            self.after(0, lambda: self._scoring_done(error))

        self.state.score_dataset_async(on_progress, on_done)

    def _progress_update(self, fraction: float, stage: str, done: int, total: int) -> None:
        self.progress.set(fraction)
        self.scoring_label.configure(
            text=f"Scoring {stage}: {done} of {total} narratives",
            text_color=theming.INK_SECONDARY,
        )

    def _scoring_done(self, error: Exception | None) -> None:
        self.score_button.configure(state="normal")
        if error is not None:
            self.scoring_label.configure(text=str(error), text_color="#e66767")
            self.set_status("Scoring failed.")
            return
        self.state.notify("scored")
        self.set_status("Scoring complete.")
