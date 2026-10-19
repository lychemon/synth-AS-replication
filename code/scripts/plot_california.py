"""
Plot California's observed and synthetic outcomes and treatment effects.

Input: data/smoking_data.csv and output/california_diagonal_donor_weights.json.
Output: output/california_diagonal_outcomes.png.
"""

import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "output")
DATA = os.path.join(ROOT, "data")
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "support"))

from data_preparation import build_california
from treatment_effects import synthetic_path, mean_effect
from outcome_plots import SyntheticSeries, plot_outcomes_and_gaps


if __name__ == "__main__":
    d = build_california(os.path.join(DATA, "smoking_data.csv"))
    with open(os.path.join(OUT, "california_diagonal_donor_weights.json")) as f:
        res = json.load(f)

    years = np.array(d["years"])
    ca = d["outcomes"]["California"]
    synth = synthetic_path(d, res["donor_weights"])

    plot_outcomes_and_gaps(
        years, ca,
        [SyntheticSeries(synth, "synthetic California", "k--", gap_style="k-")],
        treated_label="California",
        outcome_label="per-capita cigarette sales (packs)",
        treatment_year=1988,
        output_path=os.path.join(OUT, "california_diagonal_outcomes.png"),
        gap_legend=False,
    )
    print("saved output/california_diagonal_outcomes.png")
    print("mean post-1988 gap:", mean_effect(years, ca, synth, years[years > 1988]))
