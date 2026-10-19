"""
Compute diagonal log-det and analytic-center predictor weights.

Input: California and Germany CSV files in data/ and diagonal donor weights
    in output/, including leakage-free California weights.
Output: output/diagonal_predictor_weights.json.
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

from predictor_weights import diag_centers, predictor_weight_cases


def report(label, res):
    print("\n=== %s: support %s; forced zero %s; %d of %d inactive-donor slacks vanish identically"
          % (label, res["support"], res["forced_zero"], res["n_implicit_equalities"], res["n_inactive_donors"]))
    print("%-42s %10s %10s" % ("predictor", "log-det", "analytic"))
    for n in res["logdet"]:
        print("%-42s %10.4f %10.4f" % (n, res["logdet"][n], res["analytic_center"][n]))


if __name__ == "__main__":
    out = {}
    for case in predictor_weight_cases(DATA, OUT):
        out[case.tag] = diag_centers(
            np.asarray(case.X0), np.asarray(case.X1).ravel(),
            np.asarray(case.weights).ravel(), case.predictor_names)
    for label, res in out.items():
        report(label, res)
    json.dump(out, open(os.path.join(OUT, "diagonal_predictor_weights.json"), "w"), indent=1)
    print("\nwrote output/diagonal_predictor_weights.json")
