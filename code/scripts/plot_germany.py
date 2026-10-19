"""
Plot Germany's outcomes and effects using computed and ADH (2015) weights.

Input: data/german_reunification.csv, output/germany_diagonal_donor_weights.json,
    and ADH (2015) weights stored in this script.
Output: output/germany_diagonal_vs_adh_outcomes.png.
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

from data_preparation import build_germany
from treatment_effects import synthetic_path, effect_path, mean_effect
from outcome_plots import SyntheticSeries, plot_outcomes_and_gaps

ADH2015 = {"Austria": 0.42, "USA": 0.22, "Japan": 0.16,
           "Switzerland": 0.11, "Netherlands": 0.09}


if __name__ == "__main__":
    d = build_germany(os.path.join(DATA, "german_reunification.csv"))
    ours = json.load(open(os.path.join(OUT, "germany_diagonal_donor_weights.json")))["donor_weights"]
    years = np.array(d["years"])
    wg = d["outcomes"]["West Germany"]
    s_ours = synthetic_path(d, ours)
    s_adh = synthetic_path(d, ADH2015)

    plot_outcomes_and_gaps(
        years, wg,
        [
            SyntheticSeries(s_ours, "synthetic (exact)", "k--", "gap, exact"),
            SyntheticSeries(s_adh, "synthetic (ADH 2015 weights)", "k:",
                            "gap, ADH 2015 weights"),
        ],
        treated_label="West Germany",
        outcome_label="per-capita GDP (current USD)",
        treatment_year=1990,
        output_path=os.path.join(OUT, "germany_diagonal_vs_adh_outcomes.png"),
        legend_fontsize=9,
    )
    
    post = (years > 1990) & (years <= 2003)
    print("mean post-1990 gap, exact:", mean_effect(years, wg, s_ours, years[post]))
    print("mean post-1990 gap, ADH:  ", mean_effect(years, wg, s_adh, years[post]))
    pre = (years >= 1981) & (years <= 1990)
    print("validation-window RMSE, exact:",
          float(np.sqrt(np.mean(effect_path(wg, s_ours)[pre] ** 2))))
    print("validation-window RMSE, ADH:  ",
          float(np.sqrt(np.mean(effect_path(wg, s_adh)[pre] ** 2))))
