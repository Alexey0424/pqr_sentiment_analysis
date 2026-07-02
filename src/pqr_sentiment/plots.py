"""Publication grade chart library.

Design rules applied consistently across every figure:

: sentiment is a polarity, so it uses a validated diverging pair (red for
  negative, blue for positive) with a deliberate neutral gray midpoint,
: nominal magnitude comparisons use one hue, never a color per bar,
: no dual axes, two measures become stacked small multiples on one clock,
: solid hairline gridlines, recessive axes, text in ink tokens rather than
  series colors, legends whenever two or more series share a plot,
: light and dark themes are separately tuned steps of the same hues.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from .config import EMOTION_DISPLAY, SENTIMENT_DISPLAY, SENTIMENT_ORDER

__all__ = [
    "THEMES",
    "apply_theme",
    "current_theme",
    "volume_by_category",
    "sentiment_share_by_category",
    "monthly_activity",
    "probability_distributions",
    "emotion_distribution",
    "emotion_by_type_heatmap",
    "resolution_by_sentiment",
    "agreement_heatmap",
    "terms_panel",
    "save_figure",
]


@dataclass(frozen=True)
class Theme:
    """Color roles for one rendering surface."""

    surface: str
    page: str
    ink: str
    ink_secondary: str
    muted: str
    gridline: str
    baseline: str
    accent: str
    sentiment: dict[str, str]
    ramp: list[str]

    def sentiment_list(self) -> list[str]:
        return [self.sentiment[label] for label in SENTIMENT_ORDER]


LIGHT = Theme(
    surface="#fcfcfb",
    page="#f9f9f7",
    ink="#0b0b0b",
    ink_secondary="#52514e",
    muted="#898781",
    gridline="#e1e0d9",
    baseline="#c3c2b7",
    accent="#2a78d6",
    sentiment={"NEG": "#e34948", "NEU": "#a9a8a2", "POS": "#2a78d6"},
    ramp=["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)

DARK = Theme(
    surface="#1a1a19",
    page="#0d0d0d",
    ink="#ffffff",
    ink_secondary="#c3c2b7",
    muted="#898781",
    gridline="#2c2c2a",
    baseline="#383835",
    accent="#3987e5",
    sentiment={"NEG": "#e66767", "NEU": "#757470", "POS": "#3987e5"},
    ramp=["#104281", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef", "#b7d3f6", "#cde2fb"],
)

THEMES = {"light": LIGHT, "dark": DARK}

_ACTIVE: Theme = LIGHT


def apply_theme(dark: bool = False) -> Theme:
    """Configure matplotlib for one of the two validated themes."""
    global _ACTIVE
    theme = DARK if dark else LIGHT
    _ACTIVE = theme
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
            "figure.facecolor": theme.surface,
            "axes.facecolor": theme.surface,
            "savefig.facecolor": theme.surface,
            "text.color": theme.ink,
            "axes.edgecolor": theme.baseline,
            "axes.labelcolor": theme.ink_secondary,
            "axes.titlecolor": theme.ink,
            "xtick.color": theme.muted,
            "ytick.color": theme.muted,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.grid": False,
            "grid.color": theme.gridline,
            "grid.linewidth": 1.0,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
            "legend.frameon": False,
            "figure.dpi": 110,
        }
    )
    return theme


def current_theme() -> Theme:
    return _ACTIVE


def _tidy_axis(ax: plt.Axes, grid_axis: str = "x") -> None:
    """Recessive chrome: hairline grid behind the data, minimal spines."""
    ax.set_axisbelow(True)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=_ACTIVE.gridline, linewidth=1.0, linestyle="-")
    for spine in ax.spines.values():
        spine.set_color(_ACTIVE.baseline)
        spine.set_linewidth(1.0)


def _title(ax: plt.Axes, title: str, subtitle: str | None = None) -> None:
    ax.set_title(title, loc="left", fontsize=13, fontweight="semibold", color=_ACTIVE.ink, pad=14)
    if subtitle:
        ax.text(
            0, 1.02, subtitle, transform=ax.transAxes,
            fontsize=10, color=_ACTIVE.ink_secondary, va="bottom",
        )


def _display_index(values: pd.Index, display: dict | None) -> list[str]:
    if display is None:
        return [str(v) for v in values]
    return [str(display.get(v, v)) for v in values]


def volume_by_category(
    df: pd.DataFrame,
    column: str,
    title: str,
    subtitle: str | None = None,
    display: dict | None = None,
    top_n: int | None = None,
) -> plt.Figure:
    """Horizontal single hue bars for a nominal magnitude comparison."""
    counts = df[column].value_counts()
    if top_n is not None and len(counts) > top_n:
        counts = counts.head(top_n)
    counts = counts.sort_values()

    height = max(2.6, 0.42 * len(counts) + 1.4)
    fig, ax = plt.subplots(figsize=(8.6, height))
    bars = ax.barh(
        _display_index(counts.index, display),
        counts.values,
        color=_ACTIVE.accent,
        height=0.62,
    )
    for bar in bars:
        ax.text(
            bar.get_width() + counts.max() * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{int(bar.get_width()):,}",
            va="center", ha="left", fontsize=9.5, color=_ACTIVE.ink_secondary,
        )
    ax.set_xlim(0, counts.max() * 1.12)
    _tidy_axis(ax, grid_axis="x")
    _title(ax, title, subtitle)
    ax.set_xlabel("Records", fontsize=10)
    fig.tight_layout()
    return fig


def sentiment_share_by_category(
    scored: pd.DataFrame,
    cat_col: str,
    title: str,
    subtitle: str | None = None,
    display: dict | None = None,
    min_label_share: float = 0.09,
) -> plt.Figure:
    """100 percent stacked horizontal bars with the diverging sentiment triple."""
    table = pd.crosstab(scored[cat_col], scored["sentiment"], normalize="index")
    table = table.reindex(columns=SENTIMENT_ORDER, fill_value=0.0)
    counts = scored[cat_col].value_counts()
    table = table.sort_values("NEG", ascending=True)

    height = max(2.8, 0.5 * len(table) + 1.6)
    fig, ax = plt.subplots(figsize=(9.2, height))
    left = np.zeros(len(table))
    labels = _display_index(table.index, display)
    for label in SENTIMENT_ORDER:
        values = table[label].to_numpy()
        ax.barh(
            labels, values, left=left,
            color=_ACTIVE.sentiment[label],
            height=0.6,
            edgecolor=_ACTIVE.surface, linewidth=2,
            label=SENTIMENT_DISPLAY[label],
        )
        for y, (start, share) in enumerate(zip(left, values)):
            if share >= min_label_share:
                fill = _ACTIVE.sentiment[label]
                ink = "#ffffff" if label != "NEU" else _ACTIVE.ink
                if label == "NEU" and _ACTIVE is DARK:
                    ink = "#ffffff"
                ax.text(
                    start + share / 2, y, f"{share * 100:.0f}%",
                    va="center", ha="center", fontsize=9, color=ink,
                )
        left += values
    for y, name in enumerate(table.index):
        ax.text(
            1.015, y, f"n={counts[name]:,}",
            va="center", ha="left", fontsize=8.5, color=_ACTIVE.muted, transform=ax.get_yaxis_transform(),
        )
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.04), ncol=3, fontsize=9.5, handlelength=1.2)
    _tidy_axis(ax, grid_axis="")
    _title(ax, title, subtitle)
    fig.tight_layout()
    return fig


def monthly_activity(scored: pd.DataFrame, title: str, subtitle: str | None = None) -> plt.Figure:
    """Two stacked panels on one clock: monthly volume, then negative share."""
    frame = scored.dropna(subset=["received_month"]).copy()
    monthly = frame.groupby("received_month").agg(
        records=("sentiment", "size"),
        negative_share=("sentiment", lambda s: (s == "NEG").mean()),
    )
    full_index = pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS")
    monthly = monthly.reindex(full_index)

    fig, (ax_volume, ax_share) = plt.subplots(
        2, 1, figsize=(10.4, 6.2), sharex=True, height_ratios=[1.0, 1.0]
    )

    ax_volume.bar(
        monthly.index, monthly["records"].fillna(0), width=20,
        color=_ACTIVE.accent,
    )
    _tidy_axis(ax_volume, grid_axis="y")
    ax_volume.set_ylabel("Records per month", fontsize=10)
    _title(ax_volume, title, subtitle)

    known = monthly.dropna(subset=["negative_share"])
    ax_share.plot(
        known.index, known["negative_share"] * 100,
        color=_ACTIVE.sentiment["NEG"], linewidth=2, solid_joinstyle="round",
        marker="o", markersize=5.5, markerfacecolor=_ACTIVE.sentiment["NEG"],
        markeredgecolor=_ACTIVE.surface, markeredgewidth=1.6,
    )
    ax_share.set_ylim(0, 100)
    _tidy_axis(ax_share, grid_axis="y")
    ax_share.set_ylabel("Negative share, %", fontsize=10)
    ax_share.set_xlabel("Month received", fontsize=10)

    fig.align_ylabels([ax_volume, ax_share])
    fig.tight_layout()
    return fig


def probability_distributions(scored: pd.DataFrame, title: str, subtitle: str | None = None) -> plt.Figure:
    """Small multiples of the model probability mass per sentiment class."""
    columns = {"NEG": "prob_neg", "NEU": "prob_neu", "POS": "prob_pos"}
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.4), sharey=True)
    bins = np.linspace(0, 1, 21)
    for ax, (label, column) in zip(axes, columns.items()):
        color = _ACTIVE.sentiment[label]
        values = scored[column].dropna()
        ax.hist(values, bins=bins, color=color, alpha=0.16)
        ax.hist(values, bins=bins, histtype="step", color=color, linewidth=2)
        median = float(values.median())
        ax.axvline(median, color=color, linewidth=1.2)
        ax.text(
            0.97, 0.92, f"median {median:.2f}",
            transform=ax.transAxes, ha="right", fontsize=9, color=_ACTIVE.ink_secondary,
        )
        ax.set_title(SENTIMENT_DISPLAY[label], loc="left", fontsize=11, color=_ACTIVE.ink)
        ax.set_xlim(0, 1)
        _tidy_axis(ax, grid_axis="y")
    axes[0].set_ylabel("Records", fontsize=10)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="semibold", color=_ACTIVE.ink)
    if subtitle:
        fig.text(0.01, 0.9, subtitle, fontsize=10, color=_ACTIVE.ink_secondary)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig


def emotion_distribution(scored: pd.DataFrame, title: str, subtitle: str | None = None) -> plt.Figure:
    """Single hue horizontal bars: emotions are nominal here, magnitude is the job."""
    return volume_by_category(
        scored, "emotion", title, subtitle=subtitle, display=EMOTION_DISPLAY
    )


def emotion_by_type_heatmap(
    scored: pd.DataFrame,
    title: str,
    subtitle: str | None = None,
    type_display: dict | None = None,
) -> plt.Figure:
    """Heatmap of emotion share within each PQR type, one hue ramp."""
    table = pd.crosstab(scored["pqr_type"], scored["emotion"], normalize="index")
    emotion_order = scored["emotion"].value_counts().index.tolist()
    table = table.reindex(columns=emotion_order, fill_value=0.0)

    cmap = LinearSegmentedColormap.from_list("ramp", _ACTIVE.ramp)
    fig, ax = plt.subplots(figsize=(9.6, 0.72 * len(table) + 2.2))
    mesh = ax.pcolormesh(
        table.to_numpy(), cmap=cmap, vmin=0, vmax=max(0.35, table.to_numpy().max()),
        edgecolors=_ACTIVE.surface, linewidth=2,
    )
    ax.set_xticks(np.arange(len(table.columns)) + 0.5)
    ax.set_xticklabels([EMOTION_DISPLAY.get(c, c) for c in table.columns], fontsize=9.5)
    ax.set_yticks(np.arange(len(table.index)) + 0.5)
    ax.set_yticklabels(_display_index(table.index, type_display), fontsize=10)
    ax.invert_yaxis()
    for row in range(table.shape[0]):
        for col in range(table.shape[1]):
            share = table.iloc[row, col]
            r, g, b, _ = mesh.cmap(mesh.norm(share))
            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
            ink = "#ffffff" if luminance < 0.5 else "#0b0b0b"
            ax.text(col + 0.5, row + 0.5, f"{share * 100:.0f}%", ha="center", va="center", fontsize=9, color=ink)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    _title(ax, title, subtitle)
    colorbar = fig.colorbar(mesh, ax=ax, shrink=0.8, pad=0.02)
    colorbar.ax.yaxis.set_major_formatter(lambda x, _: f"{x * 100:.0f}%")
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=9, color=_ACTIVE.baseline, labelcolor=_ACTIVE.muted)
    fig.tight_layout()
    return fig


def resolution_by_sentiment(scored: pd.DataFrame, title: str, subtitle: str | None = None) -> plt.Figure:
    """Days to close per sentiment: thin horizontal boxes plus median labels."""
    frame = scored.dropna(subset=["resolution_days"])
    groups = [frame.loc[frame["sentiment"] == label, "resolution_days"] for label in SENTIMENT_ORDER]
    present = [(label, values) for label, values in zip(SENTIMENT_ORDER, groups) if len(values)]

    fig, ax = plt.subplots(figsize=(9.2, 0.8 * len(present) + 1.9))
    box = ax.boxplot(
        [values for _, values in present],
        orientation="horizontal",
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops={"linewidth": 0},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
    )
    for index, (label, values) in enumerate(present):
        color = _ACTIVE.sentiment[label]
        patch = box["boxes"][index]
        patch.set_facecolor(color)
        patch.set_alpha(0.16)
        patch.set_edgecolor(color)
        patch.set_linewidth(2)
        for part in ("whiskers", "caps"):
            for artist in box[part][index * 2 : index * 2 + 2]:
                artist.set_color(color)
        median = float(values.median())
        ax.plot([median, median], [index + 0.75, index + 1.25], color=color, linewidth=2.4)
        ax.text(
            median, index + 1.34, f"{median:.0f} d",
            ha="center", fontsize=9, color=_ACTIVE.ink_secondary,
        )
    ax.set_yticks(range(1, len(present) + 1))
    ax.set_yticklabels([SENTIMENT_DISPLAY[label] for label, _ in present], fontsize=10)
    ax.set_xlabel("Days from receipt to closure", fontsize=10)
    _tidy_axis(ax, grid_axis="x")
    _title(ax, title, subtitle)
    fig.tight_layout()
    return fig


def agreement_heatmap(
    scored: pd.DataFrame,
    title: str,
    subtitle: str | None = None,
) -> plt.Figure:
    """Confusion style grid between the two independent models."""
    table = pd.crosstab(scored["sentiment"], scored["stars_sentiment"])
    table = table.reindex(index=SENTIMENT_ORDER, columns=SENTIMENT_ORDER, fill_value=0)

    cmap = LinearSegmentedColormap.from_list("ramp", _ACTIVE.ramp)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    mesh = ax.pcolormesh(
        table.to_numpy(), cmap=cmap, vmin=0, vmax=table.to_numpy().max(),
        edgecolors=_ACTIVE.surface, linewidth=2,
    )
    ax.set_xticks(np.arange(3) + 0.5)
    ax.set_xticklabels([SENTIMENT_DISPLAY[c] for c in table.columns], fontsize=10)
    ax.set_yticks(np.arange(3) + 0.5)
    ax.set_yticklabels([SENTIMENT_DISPLAY[i] for i in table.index], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Multilingual star model", fontsize=10)
    ax.set_ylabel("Spanish sentiment model", fontsize=10)
    for row in range(3):
        for col in range(3):
            value = table.iloc[row, col]
            r, g, b, _ = mesh.cmap(mesh.norm(value))
            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
            ink = "#ffffff" if luminance < 0.5 else "#0b0b0b"
            ax.text(col + 0.5, row + 0.5, f"{value:,}", ha="center", va="center", fontsize=11, color=ink)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    _title(ax, title, subtitle)
    fig.tight_layout()
    return fig


def terms_panel(terms: pd.DataFrame, title: str, subtitle: str | None = None) -> plt.Figure:
    """Small multiples of distinctive terms per sentiment class."""
    classes = [label for label in SENTIMENT_ORDER if label in set(terms["class"])]
    fig, axes = plt.subplots(
        1, len(classes), figsize=(3.6 * len(classes) + 0.8, 4.6), sharex=False
    )
    if len(classes) == 1:
        axes = [axes]
    for ax, label in zip(axes, classes):
        subset = terms[terms["class"] == label].sort_values("score")
        color = _ACTIVE.sentiment[label]
        ax.barh(subset["term"], subset["score"], color=color, height=0.6)
        ax.set_title(SENTIMENT_DISPLAY[label], loc="left", fontsize=11, color=_ACTIVE.ink)
        ax.set_xticks([])
        _tidy_axis(ax, grid_axis="")
        ax.tick_params(axis="y", labelsize=9.5)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="semibold", color=_ACTIVE.ink)
    if subtitle:
        fig.text(0.01, 0.91, subtitle, fontsize=10, color=_ACTIVE.ink_secondary)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def save_figure(fig: plt.Figure, name: str, directory: str | Path = "reports/figures") -> Path:
    """Write a png at print resolution and return its path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    return path
