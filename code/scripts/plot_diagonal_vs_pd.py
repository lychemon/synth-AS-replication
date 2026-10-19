"""
Plot diagonal and PD outcomes and effects for California and Germany.

Input: California and Germany CSV files in data/, plus diagonal_solver_summary.json
    and pd_donor_weights.json in output/.
Output: california_diagonal_vs_pd_outcomes.png and
    germany_diagonal_vs_pd_outcomes.png in output/.
"""

import sys
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "output")
DATA = os.path.join(ROOT, "data")
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "support"))

from data_preparation import build_california, build_germany
from treatment_effects import synthetic_path
from outcome_plots import SyntheticSeries, plot_outcomes_and_gaps


if __name__ == "__main__":
    summ = json.load(open(os.path.join(OUT, "diagonal_solver_summary.json")))
    pd = json.load(open(os.path.join(OUT, "pd_donor_weights.json")))

    d = build_california(os.path.join(DATA, "smoking_data.csv"))
    years = np.array(d["years"])
    plot_outcomes_and_gaps(
        years, d["outcomes"]["California"],
        [
            SyntheticSeries(synthetic_path(d, summ["california"]["donor_weights"]),
                            "synthetic (diagonal class)", "k--", "gap, diagonal class"),
            SyntheticSeries(synthetic_path(d, pd["california"]["pd_weights"]),
                            "synthetic (full PD class)", "k:", "gap, full PD class"),
        ],
        treated_label="California",
        outcome_label="per-capita cigarette sales (packs)",
        treatment_year=1988,
        output_path=os.path.join(OUT, "california_diagonal_vs_pd_outcomes.png"),
        legend_fontsize=9,
    )

    g = build_germany(os.path.join(DATA, "german_reunification.csv"))
    years = np.array(g["years"])
    plot_outcomes_and_gaps(
        years, g["outcomes"]["West Germany"],
        [
            SyntheticSeries(synthetic_path(g, summ["germany"]["donor_weights"]),
                            "synthetic (diagonal class)", "k--", "gap, diagonal class"),
            SyntheticSeries(synthetic_path(g, pd["germany"]["pd_weights"]),
                            "synthetic (full PD class)", "k:", "gap, full PD class"),
        ],
        treated_label="West Germany",
        outcome_label="per-capita GDP (current USD)",
        treatment_year=1990,
        output_path=os.path.join(OUT, "germany_diagonal_vs_pd_outcomes.png"),
        legend_fontsize=9,
    )
    print("wrote california_diagonal_vs_pd_outcomes.png, germany_diagonal_vs_pd_outcomes.png")
