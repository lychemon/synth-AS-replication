"""
Check computed results against the paper and, when available, synthx_package.

Input: California and Germany CSV files in data/, result JSON files in output/,
    paper values stored in this script, and optionally the sibling synthx_package.
Output: output/verification.txt and a console report.
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "output")
DATA = os.path.join(ROOT, "data")
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "support"))

from data_preparation import (  # noqa: E402
    build_california,
    CA_PREDICTORS,
    build_germany,
    GER_PREDICTORS,
)
from visible_facet_solver import solve_exterior_hullfree, solve_exterior_pd  # noqa: E402
from treatment_effects import synthetic_path, mean_effect  # noqa: E402

lines = []
fails = []


def say(s=""):
    print(s); lines.append(s)


def check(label, paper, got, tol, fmt="%.4f"):
    ok = abs(got - paper) <= tol
    flag = "ok" if ok else "DIFFERS"
    if not ok:
        fails.append(label)
    say("  %-52s paper %-10s this run %-12s %s" % (label, fmt % paper, fmt % got, flag))


def load(name):
    return json.load(open(os.path.join(OUT, name)))


# ---------------------------------------------------------------------------
# Paper values
# ---------------------------------------------------------------------------
CA_W = {"Colorado": 0.160, "Connecticut": 0.068, "Montana": 0.202, "Nevada": 0.236, "Utah": 0.335}
CA_PD_W = {"Utah": 0.319, "Nevada": 0.248, "Montana": 0.302, "Colorado": 0.028, "Connecticut": 0.102}
GE_W = {"Austria": 0.507, "Switzerland": 0.166, "Japan": 0.146, "USA": 0.135, "Australia": 0.045}
GE_PD_W = {"Austria": 0.191, "Switzerland": 0.301, "Japan": 0.206, "Netherlands": 0.133,
           "Norway": 0.161, "Belgium": 0.009}
TABLE3 = [(0, 25, 34), (6, 183, 416), (29, 620, 2271), (59, 1129, 6235),
          (69, 1143, 8983), (49, 607, 6489), (14, 132, 1854)]
CA_DIAG_CENTER = [0.0, 1 / 3, 1 / 3, 0.0, 0.0, 0.0, 1 / 3]
CA_LOGDET = [0.006, 0.143, 0.144, 0.145, 0.224, 0.147, 0.191]
CA_CENTER = [0.030, 0.026, 0.036, 0.034, 0.563, 0.031, 0.279]
GE_DIAG_CENTER = [0.443, 0.134, 0.071, 0.002, 0.107, 0.243]   # analytic center of the diagonal rationalizers (Table 6, column 1)
GE_LOGDET = [0.205, 0.173, 0.150, 0.084, 0.221, 0.167]
GE_CENTER = [0.176, 0.102, 0.069, 0.125, 0.462, 0.067]

summ = load("diagonal_solver_summary.json"); pdj = load("pd_donor_weights.json")
vst = load("pd_predictor_weights_logdet.json"); anc = load("pd_predictor_weights_analytic_center.json"); dgc = load("diagonal_predictor_weights.json")

say("VERIFICATION OF THE PAPER'S NUMBERS AGAINST THIS RUN")
say("(tolerances: 0.0005 on weights and shares as printed to 3 decimals, 0.005 on")
say(" losses printed to 2 decimals, 0.5 on quantities printed as integers)")

# ---- Table 1 / D.1 California -------------------------------------------------
say("\nTable 1 and Table D.1 (California donor weights, losses, mean gaps)")
ca = summ["california"]
for k, v in CA_W.items():
    check("diagonal weight %s" % k, v, ca["donor_weights"].get(k, 0.0), 5e-4, "%.3f")
check("diagonal validation loss", 58.46, ca["validation_loss"], 5e-3, "%.2f")
for k, v in CA_PD_W.items():
    check("PD weight %s" % k, v, pdj["california"]["pd_weights"].get(k, 0.0), 5e-4, "%.3f")
check("PD validation loss", 54.70, pdj["california"]["pd_loss"], 5e-3, "%.2f")
d = build_california(os.path.join(DATA, "smoking_data.csv"))
years = np.array(d["years"])
post_years = years[(years >= 1989) & (years <= 2000)]
check("mean gap 1989-2000, diagonal (packs)", -19.0,
      mean_effect(years, d["outcomes"]["California"],
                  synthetic_path(d, ca["donor_weights"]), post_years), 0.05, "%.1f")
check("mean gap 1989-2000, PD (packs)", -19.8,
      mean_effect(years, d["outcomes"]["California"],
                  synthetic_path(d, pdj["california"]["pd_weights"]), post_years), 0.05, "%.1f")

# ---- Table 3 -------------------------------------------------------------------
say("\nTable 3 (faces by dimension: attainable / visible / all)")
for dim, (a, v, t) in enumerate(TABLE3):
    check("dim %d attainable" % dim, a, ca["faces_attainable_by_dim"][dim], 0.5, "%.0f")
    check("dim %d visible" % dim, v, ca["faces_visible_by_dim"][dim], 0.5, "%.0f")
    check("dim %d all" % dim, t, ca["faces_total_by_dim"][dim], 0.5, "%.0f")

# ---- Tables 2, 4 ---------------------------------------------------------------
say("\nTable 2 (diagonal class, log-det maximizer over the retained predictors, California)")
for k, name in enumerate(CA_PREDICTORS):
    check(name, CA_DIAG_CENTER[k], dgc["california_headline"]["logdet"][name], 5e-4, "%.4f")
say("\nTable 4 (PD class, California: log-det maximizer / analytic center)")
for k, name in enumerate(CA_PREDICTORS):
    check("log-det %s" % name, CA_LOGDET[k], vst["california_headline"]["diag"][name], 5e-4 + 1e-9, "%.3f")
    check("center  %s" % name, CA_CENTER[k], anc["california_headline"]["diag"][name], 5e-4 + 1e-9, "%.3f")
check("smallest eigenvalue of V* (x1000)", 3.7, 1000 * vst["california_headline"]["min_eig"], 0.05, "%.1f")
check("max income share over PD rationalizers", 0.038, vst["california_headline"]["ranges"][CA_PREDICTORS[0]][1], 5e-4, "%.3f")

# ---- Table 5 / D.2 Germany -----------------------------------------------------
say("\nTable 5 and Table D.2 (West Germany donor weights, losses, mean gaps)")
ge = summ["germany"]
for k, v in GE_W.items():
    check("diagonal weight %s" % k, v, ge["donor_weights"].get(k, 0.0), 5e-4, "%.3f")
check("diagonal validation loss", 45804, ge["validation_loss"], 0.5, "%.0f")
for k, v in GE_PD_W.items():
    check("PD weight %s" % k, v, pdj["germany"]["pd_weights"].get(k, 0.0), 5e-4, "%.3f")
check("PD validation loss", 20287, pdj["germany"]["pd_loss"], 0.5, "%.0f")
g = build_germany(os.path.join(DATA, "german_reunification.csv"))
ADH2015 = {"Austria": 0.42, "USA": 0.22, "Japan": 0.16, "Switzerland": 0.11, "Netherlands": 0.09}
germany_years = np.array(g["years"])
post_years = germany_years[(germany_years >= 1991) & (germany_years <= 2003)]
check("mean gap 1991-2003, exact (USD)", -1391,
      mean_effect(germany_years, g["outcomes"]["West Germany"],
                  synthetic_path(g, ge["donor_weights"]), post_years), 0.5, "%.0f")
check("mean gap 1991-2003, ADH 2015 weights (USD)", -1707,
      mean_effect(germany_years, g["outcomes"]["West Germany"],
                  synthetic_path(g, ADH2015), post_years), 0.5, "%.0f")
check("mean gap 1991-2003, PD (USD)", -1582,
      mean_effect(germany_years, g["outcomes"]["West Germany"],
                  synthetic_path(g, pdj["germany"]["pd_weights"]), post_years), 0.5, "%.0f")

# ---- Table 6 -------------------------------------------------------------------
say("\nTable 6 (West Germany: diagonal center / PD log-det maximizer / PD center)")
for k, name in enumerate(GER_PREDICTORS):
    check("diagonal %s" % name, GE_DIAG_CENTER[k], dgc["germany"]["analytic_center"][name], 5e-4 + 1e-9, "%.3f")
    check("log-det  %s" % name, GE_LOGDET[k], vst["germany"]["diag"][name], 5e-4 + 1e-9, "%.3f")
    check("center   %s" % name, GE_CENTER[k], anc["germany"]["diag"][name], 5e-4 + 1e-9, "%.3f")

say("\nRESULT: %s" % ("every reported number reproduced" if not fails
                       else "%d item(s) differ: %s" % (len(fails), "; ".join(fails))))

# ---------------------------------------------------------------------------
# Bridge comparison (the Stata command's Python backend), full precision
# ---------------------------------------------------------------------------
say("\n" + "=" * 72)
say("COMPARISON WITH THE STATA BRIDGE (synthx), full precision")
bridge = os.path.normpath(os.path.join(ROOT, "..", "synthx_package", "code", "stata", "bridge"))
if not os.path.isdir(bridge):
    say("synthx_package not found next to this package; bridge comparison skipped.")
else:
    sys.path.insert(0, bridge)
    import importlib
    sb = importlib.import_module("stata_backend")
    sys.path.insert(0, os.path.join(ROOT, "..", "synthx_package", "code", "python"))
    vr = importlib.import_module("synthx.predictor_weights")

    def load_csv(path, unitcol, cols):
        rows = list(csv.DictReader(open(path)))
        names = sorted({r[unitcol] for r in rows}); ids = {nm: i + 1 for i, nm in enumerate(names)}
        data = {"id": np.array([ids[r[unitcol]] for r in rows], float),
                "year": np.array([int(float(r["year"])) for r in rows], float)}
        for c in cols:
            data[c] = np.array([float(r[c]) if r[c] != "" else np.nan for r in rows])
        return data, names, ids

    def compare(tag, Xp1, Xp0, Zp1, Zp0, donors_p, data, names, ids, depvar, reg, specials, tr, plo, phi, xlo, xhi):
        say("\n-- %s" % tag)
        # paper pipeline at full precision: standardized X (data_preparation) + engine
        loss_p, w_p, _ = solve_exterior_hullfree(Xp0, Xp1, Zp1, Zp0)
        loss_ppd, w_ppd, _ = solve_exterior_pd(Xp0, Xp1, Zp1, Zp0)
        # bridge: raw X from the Stata-style panel + the same engine
        rb = sb.fit(data, depvar, reg, specials, "id", "year", ids[tr], plo, phi, xlo, xhi, vclass="diagonal")
        rbp = sb.fit(data, depvar, reg, specials, "id", "year", ids[tr], plo, phi, xlo, xhi, vclass="pd")
        order = [names[int(u) - 1] for u in rb["donors"]]
        assert order == list(donors_p), "donor order differs"
        say("  diagonal: loss paper-pipeline %.12f  bridge %.12f  |diff| %.2e" % (loss_p, rb["loss"], abs(loss_p - rb["loss"])))
        say("            max |w diff| %.2e ; identical to 6 decimals: %s" % (np.abs(w_p - rb["weights"]).max(),
            np.array_equal(np.round(w_p, 6), np.round(rb["weights"], 6))))
        say("  PD:       loss paper-pipeline %.12f  bridge %.12f  |diff| %.2e" % (loss_ppd, rbp["loss"], abs(loss_ppd - rbp["loss"])))
        say("            max |w diff| %.2e ; identical to 6 decimals: %s" % (np.abs(w_ppd - rbp["weights"]).max(),
            np.array_equal(np.round(w_ppd, 6), np.round(rbp["weights"], 6))))
        for metric, paper_loss, paper_weights, result in (
                ("diagonal", loss_p, w_p, rb), ("PD", loss_ppd, w_ppd, rbp)):
            if not np.isclose(paper_loss, result["loss"], rtol=1e-8, atol=1e-7):
                fails.append(tag + " " + metric + " bridge loss")
            if not np.allclose(paper_weights, result["weights"], rtol=0, atol=1e-6):
                fails.append(tag + " " + metric + " bridge weights")
        return w_p, rb

    ca_cols = ["cigsale", "lnincome", "beer", "age15to24", "retprice"]
    data, names, ids = load_csv(os.path.join(DATA, "smoking_data.csv"), "state", ca_cols)
    w_p, rb = compare("California", d["X1"], d["X0"], d["Z1"], d["Z0"], d["donors"], data, names, ids,
                      "cigsale", ["lnincome", "age15to24", "retprice"],
                      [("beer", list(range(1984, 1989))), ("cigsale", [1988]), ("cigsale", [1980]), ("cigsale", [1975])],
                      "California", 1970, 1988, 1980, 1988)
    say("  Table D.1 PD weights (Clarabel) vs bridge, max |diff| at full precision: %.1e" % max(
        abs(pdj["california"]["pd_weights"].get(names[int(u) - 1], 0.0) - float(wj))
        for u, wj in zip(rb["donors"], sb.fit(data, "cigsale", ["lnincome", "age15to24", "retprice"],
                                              [("beer", list(range(1984, 1989))), ("cigsale", [1988]), ("cigsale", [1980]), ("cigsale", [1975])],
                                              "id", "year", ids["California"], 1970, 1988, 1980, 1988, vclass="pd")["weights"])))
    # Compare saved paper predictor weights (four decimals) with full-precision bridge values.
    dg = vr.diag_rationalizers(rb["X1"], rb["X0"], rb["weights"]); pdr = vr.pd_rationalizers(rb["X1"], rb["X0"], rb["weights"])
    say("  Table 2 diagonal log-det maximizer: max |paper - bridge| = %.1e" % max(abs(dgc["california_headline"]["logdet"][n] - dg["logdet"][k]) for k, n in enumerate(CA_PREDICTORS)))
    say("  Table 2 diagonal analytic center:   max |paper - bridge| = %.1e" % max(abs(dgc["california_headline"]["analytic_center"][n] - dg["center"][k]) for k, n in enumerate(CA_PREDICTORS)))
    say("  Table 4 log-det maximizer: max |paper - bridge| = %.1e" % max(abs(vst["california_headline"]["diag"][n] - pdr["logdet_diag"][k]) for k, n in enumerate(CA_PREDICTORS)))
    say("  Table 4 analytic center:   max |paper - bridge| = %.1e" % max(abs(anc["california_headline"]["diag"][n] - pdr["center_diag"][k]) for k, n in enumerate(CA_PREDICTORS)))
    say("  identified ranges (PD):    max |paper - bridge| = %.1e" % max(max(abs(vst["california_headline"]["ranges"][n][0] - pdr["lo"][k]), abs(vst["california_headline"]["ranges"][n][1] - pdr["hi"][k])) for k, n in enumerate(CA_PREDICTORS)))

    ge_cols = ["gdp", "infrate", "trade", "schooling", "invest70", "industry"]
    data, names, ids = load_csv(os.path.join(DATA, "german_reunification.csv"), "country", ge_cols)
    w_p, rb = compare("West Germany", g["X1"], g["X0"], g["Z1"], g["Z0"], g["donors"], data, names, ids,
                      "gdp", ["gdp", "trade", "infrate", "industry"], [("schooling", [1970, 1975]), ("invest70", [1980])],
                      "West Germany", 1981, 1990, 1971, 1980)
    dg = vr.diag_rationalizers(rb["X1"], rb["X0"], rb["weights"]); pdr = vr.pd_rationalizers(rb["X1"], rb["X0"], rb["weights"])
    say("  Table 6 diagonal log-det maximizer: max |paper - bridge| = %.1e" % max(abs(dgc["germany"]["logdet"][n] - dg["logdet"][k]) for k, n in enumerate(GER_PREDICTORS)))
    say("  Table 6 diagonal analytic center (column 1 of the paper): max |paper - bridge| = %.1e" % max(abs(dgc["germany"]["analytic_center"][n] - dg["center"][k]) for k, n in enumerate(GER_PREDICTORS)))
    say("  Table 6 log-det maximizer: max |paper - bridge| = %.1e" % max(abs(vst["germany"]["diag"][n] - pdr["logdet_diag"][k]) for k, n in enumerate(GER_PREDICTORS)))
    say("  Table 6 analytic center:   max |paper - bridge| = %.1e" % max(abs(anc["germany"]["diag"][n] - pdr["center_diag"][k]) for k, n in enumerate(GER_PREDICTORS)))

with open(os.path.join(OUT, "verification.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("\nwritten to output/verification.txt")
sys.exit(1 if fails else 0)
