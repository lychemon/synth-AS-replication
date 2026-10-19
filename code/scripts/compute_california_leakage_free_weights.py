"""
Compute California donor weights with 1975, 1980, and 1988 excluded from validation.

Input: data/smoking_data.csv.
Output: output/california_diagonal_donor_weights_leakage_free.json.
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

from data_preparation import build_california, CA_HOLDOUT_IDX
from visible_facet_solver import solve_exterior_hullfree

if __name__ == "__main__":
    d = build_california(os.path.join(DATA, "smoking_data.csv"))
    X1, X0 = d["X1"], d["X0"]
    Zt1 = d["Z1"][CA_HOLDOUT_IDX]
    Zt0 = d["Z0"][CA_HOLDOUT_IDX, :]

    loss, w, _ = solve_exterior_hullfree(X0, X1, Zt1, Zt0)
    out = dict(loss=float(loss),
               donor_weights={d["donors"][i]: float(w[i])
                              for i in np.flatnonzero(w > 0)})
    with open(os.path.join(OUT, "california_diagonal_donor_weights_leakage_free.json"), "w") as f:
        json.dump(out, f, indent=2)
    displayed = dict(out, donor_weights={name: round(weight, 4)
                                        for name, weight in out["donor_weights"].items()})
    print(json.dumps(displayed, indent=2))
