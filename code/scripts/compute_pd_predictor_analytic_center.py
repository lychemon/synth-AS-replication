"""
Compute PD analytic-center predictor weights for California and Germany.

Input: California and Germany CSV files in data/, diagonal donor weights in
    output/ (including leakage-free California), and
    output/pd_predictor_weights_logdet.json.
Output: output/pd_predictor_weights_analytic_center.json.
"""

import sys
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "output")
DATA = os.path.join(ROOT, "data")
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "support"))

from predictor_weights import analytic_center, predictor_weight_cases, verify


def run_case(tag, X1, X0, w, names, vstar_diag, results):
    Vc, out = analytic_center(X1, X0, w, names)
    out["w_check_max_diff"] = verify(X1, X0, Vc, w)
    out["vstar_diag_for_comparison"] = vstar_diag
    results[tag] = out


if __name__ == "__main__":
    vstar_res = json.load(open(os.path.join(OUT, "pd_predictor_weights_logdet.json")))
    results = {}

    for case in predictor_weight_cases(DATA, OUT):
        run_case(case.tag, case.X1, case.X0, case.weights,
                 case.predictor_names, vstar_res[case.tag]["diag"], results)

    for name, res in results.items():
        print(f"\n=== {name} ===  status {res['status']}  "
              f"min eig {res['min_eig']:.2e}  "
              f"w-check {res['w_check_max_diff']:.2e}  "
              f"implicit eqs {res['n_implicit_equalities']}/"
              f"{res['n_inactive']}")
        for k in res["diag"]:
            print(f"  {k:<40} center = {res['diag'][k]:7.4f}   "
                  f"logdet max = {res['vstar_diag_for_comparison'][k]:7.4f}")

    with open(os.path.join(OUT, "pd_predictor_weights_analytic_center.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nwritten to output/pd_predictor_weights_analytic_center.json")
