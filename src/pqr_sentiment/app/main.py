"""PQR Sentiment Studio: entry point and window shell.

Run with ``pqr_studio`` (console script) or ``python -m pqr_sentiment.app``.
"""

from __future__ import annotations

import customtkinter as ctk
import matplotlib

matplotlib.use("Agg")  # figures are rendered into Tk canvases explicitly

from . import theming
from .state import AppState
from .views.dashboard import DashboardView
from .views.data import DataView
from .views.export import ExportView
from .views.quick import QuickView
from .widgets import body_font, heading_font

NAV = [
    ("Quick Analysis", "quick"),
    ("Data", "data"),
    ("Dashboard", "dashboard"),
    ("Export", "export"),
]


class StudioApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__(fg_color=theming.PAGE)
        self.title("PQR Sentiment Studio")
        self.geometry("1280x820")
        self.minsize(1080, 700)

        self.state_object = AppState()

        self.sidebar = ctk.CTkFrame(self, width=224, fg_color=theming.SURFACE, corner_radius=0)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        ctk.CTkLabel(
            self.sidebar, text="PQR Sentiment\nStudio", justify="left",
            font=heading_font(19), text_color=theming.INK, anchor="w",
        ).pack(fill="x", padx=20, pady=(24, 4))
        ctk.CTkLabel(
            self.sidebar, text="Citizen feedback analytics\nwith Spanish transformers",
            justify="left", font=body_font(11), text_color=theming.MUTED, anchor="w",
        ).pack(fill="x", padx=20, pady=(0, 18))

        self.nav_buttons: dict[str, ctk.CTkButton] = {}
        for label, key in NAV:
            button = ctk.CTkButton(
                self.sidebar, text=label, anchor="w", height=38, corner_radius=8,
                fg_color="transparent", hover_color=theming.CARD,
                text_color=theming.INK_SECONDARY, font=body_font(13),
                command=lambda k=key: self.show(k),
            )
            button.pack(fill="x", padx=12, pady=2)
            self.nav_buttons[key] = button

        ctk.CTkLabel(
            self.sidebar,
            text="Models run locally.\nNo narrative leaves this machine.",
            justify="left", font=body_font(10), text_color=theming.MUTED, anchor="w",
        ).pack(side="bottom", fill="x", padx=20, pady=16)

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(side="left", fill="both", expand=True)

        self.content = ctk.CTkScrollableFrame(body, fg_color="transparent")
        self.content.pack(fill="both", expand=True, padx=22, pady=(18, 6))

        self.status = ctk.CTkLabel(
            body, text="Ready.", anchor="w", font=body_font(11), text_color=theming.MUTED,
        )
        self.status.pack(fill="x", padx=24, pady=(0, 8))

        self.views = {
            "quick": QuickView(self.content, self.state_object, self.set_status),
            "data": DataView(self.content, self.state_object, self.set_status),
            "dashboard": DashboardView(self.content, self.state_object, self.set_status),
            "export": ExportView(self.content, self.state_object, self.set_status),
        }
        self.active: str | None = None
        self.show("quick")

    def set_status(self, message: str) -> None:
        self.status.configure(text=message)

    def show(self, key: str) -> None:
        if self.active == key:
            return
        for view in self.views.values():
            view.pack_forget()
        self.views[key].pack(fill="both", expand=True)
        for nav_key, button in self.nav_buttons.items():
            selected = nav_key == key
            button.configure(
                fg_color=theming.CARD if selected else "transparent",
                text_color=theming.INK if selected else theming.INK_SECONDARY,
            )
        self.active = key


def run() -> None:
    ctk.set_appearance_mode("dark")
    app = StudioApp()
    app.mainloop()


if __name__ == "__main__":
    run()
