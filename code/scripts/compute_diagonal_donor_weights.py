"""
Compute diagonal donor weights, losses, and face counts for California and Germany.

Input: data/smoking_data.csv and data/german_reunification.csv.
Output: Donor-weight JSON files, diagonal_solver_summary.json, and face-count
    CSV files in output/.
"""

import itertools
import json
import os
import sys
import time
from collections import Counter

import numpy as np
from scipy.spatial import ConvexHull


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "output")
DATA = os.path.join(ROOT, "data")
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "support"))

from data_preparation import build_california, build_germany  # noqa: E402
from visible_facet_solver import (visible_facets, feasible_pieces,  # noqa: E402
                             solve_exterior_hullfree, LP_COUNT)


def qhull_facets(X0, tol=1e-9):
    """
    Deduplicated facets of conv(X0) as (frozenset(vertices), a, b) with
    a.x + b <= 0 inside (qhull convention).
    """
    hull = ConvexHull(X0.T)
    eqs = {}
    for eq in hull.equations:
        eqs[tuple(np.round(eq, 9))] = np.array(eq)
    out = []
    pts = X0.T
    for eq in eqs.values():
        a, b = eq[:-1], eq[-1]
        on = np.where(np.abs(pts @ a + b) <= tol * max(1.0, abs(b)))[0]
        out.append((frozenset(int(i) for i in on), a, b))
    return out, hull


def faces_below(facet_sets):
    faces = set()
    for fs in facet_sets:
        verts = sorted(fs)
        for k in range(1, len(verts) + 1):
            for sub in itertools.combinations(verts, k):
                faces.add(frozenset(sub))
    return faces


def counts_by_dim(faces, K):
    c = Counter(len(f) - 1 for f in faces)
    return [int(c.get(d, 0)) for d in range(K)]


def run_application(name, X1, X0, Z1, Z0, donors):
    """
    Report face counts and find donor weights using the paper's search.

    The exhaustive sweep checks every visible face to count feasible sign
    regions. It checks feasibility only; it does not minimize validation loss.
    The branch-and-bound search then finds the weights independently.

    Return the report and a weight for each donor, in the order of donors.
    """
    K, J = X0.shape
    rep = {"name": name, "K": K, "J": J}
    t0 = time.time()

    # Compare the visible facets found by Qhull and the facet walk.
    facets, hull = qhull_facets(X0)
    vis_q = {fs for fs, a, b in facets if a @ X1 + b > 1e-9}
    vf = visible_facets(X0, X1)                    
    vis_g = set(vf.keys())
    rep["n_facets_total"] = len(facets)
    rep["n_visible_facets_qhull"] = len(vis_q)
    rep["n_visible_facets_giftwrap"] = len(vis_g)
    rep["visible_sets_agree"] = sorted(map(sorted, vis_q)) == sorted(map(sorted, vis_g))
    print(f"[{name}] facets={len(facets)} visible: qhull={len(vis_q)} "
          f"giftwrap={len(vis_g)} agree={rep['visible_sets_agree']}", flush=True)

    # Count all hull faces and those belonging to visible facets.
    all_f = faces_below([fs for fs, _, _ in facets])
    vis_f = faces_below(vis_q)
    rep["faces_total_by_dim"] = counts_by_dim(all_f, K)
    rep["faces_visible_by_dim"] = counts_by_dim(vis_f, K)
    print(f"[{name}] total faces={len(all_f)} visible faces={len(vis_f)}", flush=True)

    # Exhaustive sweep for the report: check every visible face without pruning.
    # Count its nonempty closed sign regions, without optimizing over them.
    sweep_start = time.time()
    LP_COUNT[0] = 0
    attainable_faces = []
    n_pieces = 0
    for i, face in enumerate(sorted(vis_f, key=len, reverse=True)):
        # Reuse a facet's normal when it is already available.
        normal = None
        fn = vf.get(face)
        if fn is not None:
            normal = fn[0]
        pieces = feasible_pieces(face, X0, X1, normal)
        if pieces:
            attainable_faces.append(face)
            n_pieces += len(pieces)
        if (i + 1) % 500 == 0:
            print(f"[{name}]   face sweep {i+1}/{len(vis_f)} "
                  f"({time.time()-sweep_start:.0f}s, LPs={LP_COUNT[0]})", flush=True)
    rep["faces_attainable_by_dim"] = counts_by_dim(set(attainable_faces), K)
    rep["n_pieces"] = int(n_pieces)
    rep["sweep_lps"] = int(LP_COUNT[0])
    rep["sweep_seconds"] = round(time.time() - sweep_start, 1)
    print(f"[{name}] face sweep: attainable={rep['faces_attainable_by_dim']} "
          f"pieces={n_pieces} "
          f"({rep['sweep_seconds']}s)", flush=True)

    # Run the paper's branch-and-bound search independently of the sweep.
    # It checks regions as needed and skips faces that cannot improve the loss.
    search_start = time.time()
    loss, weights, info = solve_exterior_hullfree(X0, X1, Z1, Z0)
    rep["validation_loss"] = float(loss)
    rep["donor_weights"] = {donors[j]: float(weights[j])
                            for j in range(J) if weights[j] > 0}
    rep["branch_and_bound_faces_touched"] = int(info["faces_touched"])
    # Measure the search separately; total time includes the checks and sweep.
    rep["branch_and_bound_seconds"] = round(time.time() - search_start, 1)
    rep["total_seconds"] = round(time.time() - t0, 1)
    print(f"[{name}] branch-and-bound optimum: loss={loss:.6f} "
          f"faces_touched={info['faces_touched']} "
          f"({rep['branch_and_bound_seconds']}s)", flush=True)
    return rep, weights


def write_face_counts(rep, path):
    K = rep["K"]
    with open(path, "w") as f:
        f.write("dimension,attainable,visible,total\n")
        for d in range(K):
            f.write("%d,%d,%d,%d\n" % (
                d, rep["faces_attainable_by_dim"][d],
                rep["faces_visible_by_dim"][d], rep["faces_total_by_dim"][d]))


def main():
    os.makedirs(OUT, exist_ok=True)
    summary = {}

    # ---- California ----
    d = build_california(os.path.join(DATA, "smoking_data.csv"))
    rep, _ = run_application("california", d["X1"], d["X0"],
                             d["Z1"], d["Z0"], d["donors"])
    summary["california"] = rep
    write_face_counts(rep, os.path.join(OUT, "california_face_counts.csv"))
    with open(os.path.join(OUT, "california_diagonal_donor_weights.json"), "w") as f:
        json.dump({"donor_weights": rep["donor_weights"],
                   "validation_loss": rep["validation_loss"]}, f, indent=1)

    # ---- Germany ----
    g = build_germany(os.path.join(DATA, "german_reunification.csv"))
    repg, _ = run_application("germany", g["X1"], g["X0"],
                              g["Z1"], g["Z0"], g["donors"])
    summary["germany"] = repg
    write_face_counts(repg, os.path.join(OUT, "germany_face_counts.csv"))
    with open(os.path.join(OUT, "germany_diagonal_donor_weights.json"), "w") as f:
        json.dump({"donor_weights": repg["donor_weights"],
                   "validation_loss": repg["validation_loss"]}, f, indent=1)

    with open(os.path.join(OUT, "diagonal_solver_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("\nALL DONE. Summary written to output/diagonal_solver_summary.json", flush=True)


if __name__ == "__main__":
    main()
