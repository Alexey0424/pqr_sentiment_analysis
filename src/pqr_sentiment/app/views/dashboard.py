"""Dashboard: KPI row plus tabbed dark charts over the scored dataset."""

from __future__ import annotations

import customtkinter as ctk

from ... import config, plots
from ...text_mining import top_terms_by_class
from .. import theming
from ..state import AppState
from ..widgets import ChartHost, KpiCard, body_font, heading_font


class DashboardView(ctk.CTkFrame):
    def __init__(self, master, state: AppState, set_status) -> None:
        super().__init__(master, fg_color="transparent")
        self.state = state
        self.set_status = set_status

        ctk.CTkLabel(
            self, text="Dashboard", font=heading_font(22), text_color=theming.INK, anchor="w",
        ).pack(fill="x", pady=(0, 2))
        self.subtitle = ctk.CTkLabel(
            self, text="Score a dataset in the Data view to populate this dashboard.",
            font=body_font(13), text_color=theming.MUTED, anchor="w",
        )
        self.subtitle.pack(fill="x", pady=(0, 12))

        self.kpi_row = ctk.CTkFrame(self, fg_color="transparent")
        self.kpi_row.pack(fill="x")
        self.kpis = {
            "records": KpiCard(self.kpi_row, "Records"),
            "negative": KpiCard(self.kpi_row, "Negative share"),
            "route": KpiCard(self.kpi_row, "Most negative entry route"),
            "days": KpiCard(self.kpi_row, "Median days to close"),
        }
        for index, card in enumerate(self.kpis.values()):
            card.grid(row=0, column=index, sticky="ew", padx=(0 if index == 0 else 10, 0))
            self.kpi_row.columnconfigure(index, weight=1)

        self.tabs = ctk.CTkSegmentedButton(
            self,
            values=["Sentiment", "Geography", "Timeline", "Emotions", "Terms"],
            command=self.render_tab,
            font=body_font(13),
            selected_color=theming.ACCENT,
            selected_hover_color=theming.ACCENT_HOVER,
            unselected_color=theming.CARD,
            unselected_hover_color=theming.CARD_BORDER,
            fg_color=theming.CARD,
        )
        self.tabs.pack(fill="x", pady=(14, 10))
        self.tabs.set("Sentiment")

        self.chart = ChartHost(self)
        self.chart.pack(fill="both", expand=True)

        self.state.subscribe(self.on_state_event)

    # ------------------------------------------------------------------
    def on_state_event(self, event: str) -> None:
        if event == "scored" and self.state.scored is not None:
            self.render_kpis()
            self.render_tab(self.tabs.get())

    def render_kpis(self) -> None:
        scored = self.state.scored
        negative = (scored["sentiment"] == "NEG").mean()
        route_shares = (
            scored.groupby("entry_route")["sentiment"].apply(lambda s: (s == "NEG").mean())
        )
        worst_route = route_shares.idxmax()
        median_days = scored["resolution_days"].median()
        self.kpis["records"].set(f"{len(scored):,}")
        self.kpis["negative"].set(f"{negative * 100:.0f}%")
        self.kpis["route"].set(
            f"{config.VALUE_DISPLAY.get(worst_route, worst_route)} "
            f"({route_shares.max() * 100:.0f}%)"
        )
        self.kpis["days"].set("..." if median_days != median_days else f"{median_days:.0f}")
        self.subtitle.configure(
            text=f"{self.state.source_name}: scored with the Spanish sentiment and emotion models."
        )

    def render_tab(self, name: str) -> None:
        scored = self.state.scored
        if scored is None:
            return
        plots.apply_theme(dark=True)
        try:
            if name == "Sentiment":
                figure = plots.sentiment_share_by_category(
                    scored, "pqr_type", "Sentiment by record type",
                    subtitle="Share of sentiment labels within each PQR type",
                    display=config.VALUE_DISPLAY,
                )
            elif name == "Geography":
                figure = plots.sentiment_share_by_category(
                    scored, "municipality", "Sentiment by municipality",
                    subtitle="Share of sentiment labels within each municipality",
                )
            elif name == "Timeline":
                figure = plots.monthly_activity(
                    scored, "Monthly volume and negative share",
                    subtitle="Records per month and the share of negative narratives",
                )
            elif name == "Emotions":
                figure = plots.emotion_distribution(
                    scored, "Dominant emotion per record",
                    subtitle="The formal register hides emotion in most narratives",
                )
            else:
                terms = top_terms_by_class(scored, config.TEXT_COL, "sentiment", top_n=10)
                figure = plots.terms_panel(
                    terms, "Distinctive terms per sentiment",
                    subtitle="Highest mean TF IDF terms within each class",
                )
            self.chart.show(figure)
        except Exception as error:  # noqa: BLE001 - shown in the UI
            self.set_status(f"Chart failed: {error}")
