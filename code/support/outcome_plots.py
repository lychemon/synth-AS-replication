"""
Plot actual and synthetic outcomes alongside their gaps.
"""

from dataclasses import dataclass
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from treatment_effects import effect_path


@dataclass
class SyntheticSeries:
    """
    Synthetic outcome, with its outcome and gap display settings. 
    """
    outcome: object
    label: str
    style: str
    gap_label: Optional[str] = None
    gap_style: Optional[str] = None


def plot_outcomes_and_gaps(years, treated, synthetic_series, *, treated_label,
                           outcome_label, treatment_year, output_path,
                           legend_fontsize=None, gap_legend=True):
    """
    Save actual/synthetic outcomes and their gaps in adjacent panels.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.plot(years, treated, "k-", label=treated_label)
    for series in synthetic_series:
        ax.plot(years, series.outcome, series.style, label=series.label)
    # Stata xline: gs6, medium, dash.
    ax.axvline(treatment_year, color="#606060", lw=0.9, ls=(0, (9, 4.5)))
    ax.set_xlabel("year")
    ax.set_ylabel(outcome_label)
    legend_options = {"frameon": False}
    if legend_fontsize is not None:
        legend_options["fontsize"] = legend_fontsize
    ax.legend(**legend_options)

    ax = axes[1]
    for series in synthetic_series:
        gap_options = {}
        if series.gap_label is not None:
            gap_options["label"] = series.gap_label
        ax.plot(years, effect_path(treated, series.outcome),
                series.gap_style or series.style, **gap_options)
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(treatment_year, color="#606060", lw=0.9, ls=(0, (9, 4.5)))
    ax.set_xlabel("year")
    ax.set_ylabel("gap (treated - synthetic)")
    if gap_legend:
        ax.legend(**legend_options)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
