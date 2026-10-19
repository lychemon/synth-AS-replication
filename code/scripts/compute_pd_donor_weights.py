"""
Compute PD donor weights, losses, and effects for California and Germany.

Input: California and Germany CSV files in data/ and
    output/diagonal_solver_summary.json for comparison.
Output: output/pd_donor_weights.json.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "output")
DATA = os.path.join(ROOT, "data")
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "support"))

from data_preparation import build_california, build_germany  # noqa: E402
from visible_facet_solver import solve_exterior_pd           # noqa: E402
from treatment_effects import synthetic_path, mean_effect    # noqa: E402


def pd_optimum(name, X0, X1, Z1, Z0, donors):
    """Run the PD solver and format its results for the output file."""
    loss, weights, info = solve_exterior_pd(X0, X1, Z1, Z0)
    return {
        "name": name,
        "n_visible_facets": info["n_visible_facets"],
        "pd_loss": loss,
        "pd_weights": {donors[j]: float(weights[j])
                       for j in range(len(donors)) if weights[j] > 0},
        "facet_vertices": [donors[j] for j in info["facet_vertices"]],
    }


if __name__ == "__main__":
    report = {}

    d = build_california(os.path.join(DATA, "smoking_data.csv"))
    ca = pd_optimum("california", d["X0"], d["X1"], d["Z1"], d["Z0"], d["donors"])
    ca["post_effect_1989_2000"] = mean_effect(
        d["years"], d["outcomes"]["California"],
        synthetic_path(d, ca["pd_weights"]), range(1989, 2001))
    report["california"] = ca

    g = build_germany(os.path.join(DATA, "german_reunification.csv"))
    ge = pd_optimum("germany", g["X0"], g["X1"], g["Z1"], g["Z0"], g["donors"])
    ge["post_effect_1991_2003"] = mean_effect(
        g["years"], g["outcomes"]["West Germany"],
        synthetic_path(g, ge["pd_weights"]), range(1991, 2004))
    report["germany"] = ge

    with open(os.path.join(OUT, "diagonal_solver_summary.json")) as f:
        diag = json.load(f)
    for k in ("california", "germany"):
        report[k]["diagonal_loss"] = diag[k]["validation_loss"]
        report[k]["diagonal_weights"] = diag[k]["donor_weights"]

    with open(os.path.join(OUT, "pd_donor_weights.json"), "w") as f:
        json.dump(report, f, indent=1)

    for k in ("california", "germany"):
        r = report[k]
        print(f"== {k} ==")
        print(f"  visible facets: {r['n_visible_facets']}")
        print(f"  PD-class optimum loss:      {r['pd_loss']:.6f}")
        print(f"  diagonal-class optimum loss:{r['diagonal_loss']:.6f}")
        for label, key in (("PD", "pd_weights"), ("diagonal", "diagonal_weights")):
            displayed = {name: round(weight, 6) for name, weight in r[key].items()
                         if weight > 1e-6}
            print(f"  {label} weights: {displayed}")
        eff = [v for kk, v in r.items() if kk.startswith('post_effect')]
        print(f"  PD post-treatment mean effect: {eff[0]:.1f}")
        print()
