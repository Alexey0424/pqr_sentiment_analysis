"""Reusable interface building blocks."""

from __future__ import annotations

import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from . import theming


def heading_font(size: int = 20, weight: str = "bold") -> ctk.CTkFont:
    return ctk.CTkFont(family=theming.FONT_FAMILY, size=size, weight=weight)


def body_font(size: int = 13, weight: str = "normal") -> ctk.CTkFont:
    return ctk.CTkFont(family=theming.FONT_FAMILY, size=size, weight=weight)


class KpiCard(ctk.CTkFrame):
    """A stat tile: sentence case label above a large proportional value."""

    def __init__(self, master, label: str, **kwargs) -> None:
        super().__init__(
            master, fg_color=theming.CARD, corner_radius=12,
            border_width=1, border_color=theming.CARD_BORDER, **kwargs,
        )
        self._label = ctk.CTkLabel(
            self, text=label, font=body_font(12), text_color=theming.MUTED, anchor="w",
        )
        self._label.pack(fill="x", padx=16, pady=(12, 0))
        self._value = ctk.CTkLabel(
            self, text="...", font=heading_font(26), text_color=theming.INK, anchor="w",
        )
        self._value.pack(fill="x", padx=16, pady=(0, 12))

    def set(self, value: str) -> None:
        self._value.configure(text=value)


class ProbabilityRow(ctk.CTkFrame):
    """A labeled horizontal probability bar in a sentiment or accent color."""

    def __init__(self, master, label: str, color: str, **kwargs) -> None:
        super().__init__(master, fg_color="transparent", **kwargs)
        self.columnconfigure(1, weight=1)
        self._name = ctk.CTkLabel(
            self, text=label, width=86, anchor="w",
            font=body_font(13), text_color=theming.INK_SECONDARY,
        )
        self._name.grid(row=0, column=0, sticky="w")
        self._bar = ctk.CTkProgressBar(
            self, height=12, corner_radius=6,
            progress_color=color, fg_color=theming.CARD_BORDER,
        )
        self._bar.grid(row=0, column=1, sticky="ew", padx=(8, 10))
        self._bar.set(0)
        self._value = ctk.CTkLabel(
            self, text="", width=52, anchor="e",
            font=body_font(13, "bold"), text_color=theming.INK,
        )
        self._value.grid(row=0, column=2, sticky="e")

    def set(self, share: float) -> None:
        self._bar.set(max(0.0, min(1.0, share)))
        self._value.configure(text=f"{share * 100:.1f}%")

    def reset(self) -> None:
        self._bar.set(0)
        self._value.configure(text="")


class ChartHost(ctk.CTkFrame):
    """Embeds a matplotlib figure and disposes of it cleanly.

    The attribute is deliberately not called ``_canvas``: CTkFrame owns an
    internal ``_canvas`` and shadowing it breaks the frame's own redraws.
    """

    def __init__(self, master, **kwargs) -> None:
        super().__init__(master, fg_color=theming.SURFACE, corner_radius=12, **kwargs)
        self._figure_canvas: FigureCanvasTkAgg | None = None

    def show(self, figure: Figure) -> None:
        self.clear()
        self._figure_canvas = FigureCanvasTkAgg(figure, master=self)
        self._figure_canvas.draw()
        widget = self._figure_canvas.get_tk_widget()
        widget.configure(background=theming.SURFACE, highlightthickness=0)
        widget.pack(fill="both", expand=True, padx=6, pady=6)

    def clear(self) -> None:
        if self._figure_canvas is not None:
            import matplotlib.pyplot as plt

            widget = self._figure_canvas.get_tk_widget()
            plt.close(self._figure_canvas.figure)
            self._figure_canvas = None
            widget.destroy()


class Section(ctk.CTkFrame):
    """A titled card used to group content inside a view."""

    def __init__(self, master, title: str, subtitle: str | None = None, **kwargs) -> None:
        super().__init__(
            master, fg_color=theming.CARD, corner_radius=12,
            border_width=1, border_color=theming.CARD_BORDER, **kwargs,
        )
        ctk.CTkLabel(
            self, text=title, font=heading_font(15, "bold"),
            text_color=theming.INK, anchor="w",
        ).pack(fill="x", padx=16, pady=(14, 0 if subtitle else 10))
        if subtitle:
            ctk.CTkLabel(
                self, text=subtitle, font=body_font(12),
                text_color=theming.MUTED, anchor="w", justify="left", wraplength=760,
            ).pack(fill="x", padx=16, pady=(2, 10))
