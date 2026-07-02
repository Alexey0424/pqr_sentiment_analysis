"""Export: scored csv plus a png chart pack to a chosen folder."""

from __future__ import annotations

import threading
from tkinter import filedialog

import customtkinter as ctk

from .. import theming
from ..state import AppState
from ..widgets import Section, body_font, heading_font


class ExportView(ctk.CTkFrame):
    def __init__(self, master, state: AppState, set_status) -> None:
        super().__init__(master, fg_color="transparent")
        self.state = state
        self.set_status = set_status

        ctk.CTkLabel(
            self, text="Export", font=heading_font(22), text_color=theming.INK, anchor="w",
        ).pack(fill="x", pady=(0, 2))
        ctk.CTkLabel(
            self,
            text="Write the scored dataset and a chart pack, ready for a report or a slide deck.",
            font=body_font(13), text_color=theming.MUTED, anchor="w",
        ).pack(fill="x", pady=(0, 12))

        card = Section(
            self, "Chart pack",
            "pqr_scored.csv plus five print resolution charts: sentiment by type and by "
            "municipality, the monthly timeline, emotions and distinctive terms.",
        )
        card.pack(fill="x")
        self.export_button = ctk.CTkButton(
            card, text="Choose folder and export...", width=230, height=34, corner_radius=8,
            fg_color=theming.ACCENT, hover_color=theming.ACCENT_HOVER,
            font=body_font(13, "bold"), command=self.export,
        )
        self.export_button.pack(anchor="w", padx=16, pady=(0, 10))
        self.result_label = ctk.CTkLabel(
            card, text="Score a dataset first.", font=body_font(12),
            text_color=theming.MUTED, anchor="w", justify="left",
        )
        self.result_label.pack(fill="x", padx=16, pady=(0, 14))

        self.state.subscribe(self.on_state_event)

    def on_state_event(self, event: str) -> None:
        if event == "scored" and self.state.scored is not None:
            self.result_label.configure(
                text="Scores are ready to export.", text_color=theming.INK_SECONDARY,
            )

    def export(self) -> None:
        if self.state.scored is None:
            self.result_label.configure(text="Nothing to export yet: score a dataset first.")
            return
        directory = filedialog.askdirectory(title="Choose an export folder")
        if not directory:
            return
        self.export_button.configure(state="disabled", text="Exporting...")
        self.set_status("Exporting chart pack...")

        def work() -> None:
            try:
                written = self.state.export(directory)
                message = f"Wrote {len(written)} files to {directory}"
                self.after(0, lambda: self._done(message, ok=True))
            except Exception as error:  # noqa: BLE001 - shown in the UI
                self.after(0, lambda: self._done(str(error), ok=False))

        threading.Thread(target=work, daemon=True).start()

    def _done(self, message: str, ok: bool) -> None:
        self.export_button.configure(state="normal", text="Choose folder and export...")
        self.result_label.configure(
            text=message, text_color=theming.INK_SECONDARY if ok else "#e66767",
        )
        self.set_status("Export complete." if ok else "Export failed.")
