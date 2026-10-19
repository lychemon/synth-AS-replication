"""
Compute PD log-det predictor weights and allowable predictor-weight ranges.

Input: California and Germany CSV files in data/ and diagonal donor weights
    in output/, including leakage-free California weights.
Output: output/pd_predictor_weights_logdet.json.
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

from predictor_weights import (
    compute_pd_logdet_and_ranges,
    predictor_weight_cases,
    verify,
)


if __name__ == "__main__":
    results = {}

    labels = {"california_headline": "headline",
              "california_leakage_free": "leakage-free",
              "germany": "germany"}
    for case in predictor_weight_cases(DATA, OUT):
        print(f"{labels[case.tag]} w* check vs table:",
              {case.donors[i]: round(float(case.weights[i]), 4)
               for i in np.flatnonzero(case.weights > 0)})
        V, out = compute_pd_logdet_and_ranges(
            case.X1, case.X0, case.weights, case.predictor_names)
        out["w_check_max_diff"] = verify(case.X1, case.X0, V, case.weights)
        results[case.tag] = out

    for name, res in results.items():
        print(f"\n=== {name} ===  status {res['status']}  "
              f"min eig {res['min_eig']:.2e}  "
              f"w-check {res['w_check_max_diff']:.2e}")
        for k, v in res["diag"].items():
            print(f"  {k:<40} V*_kk = {v:7.4f}   range {res['ranges'][k]}")

    with open(os.path.join(OUT, "pd_predictor_weights_logdet.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nwritten to output/pd_predictor_weights_logdet.json")
