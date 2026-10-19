"""
Run all replication computations, plots, and verification in order.

Input: California and Germany CSV files in data/ and the scripts in this folder.
Output: Result JSON files, face-count CSV files, figures, verification.txt, and
    run_all.log in output/.
"""
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "output")

os.makedirs(OUT, exist_ok=True)
os.chdir(HERE)

STEPS = [
    "compute_diagonal_donor_weights.py",
    "plot_california.py",
    "plot_germany.py",
    "compute_pd_donor_weights.py",
    "plot_diagonal_vs_pd.py",
    "compute_california_leakage_free_weights.py",
    "compute_pd_predictor_logdet.py",
    "compute_pd_predictor_analytic_center.py",
    "compute_diagonal_predictor_weights.py",
    "verify_results.py",
]

log = open(os.path.join(OUT, "run_all.log"), "w", encoding="utf-8")


def say(s=""):
    print(s, flush=True)
    log.write(s + "\n"); log.flush()


say("run_all.py  %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
say("python %s on %s" % (platform.python_version(), platform.platform()))
for pkg in ("numpy", "scipy", "cvxpy", "clarabel", "matplotlib"):
    try:
        m = __import__(pkg)
        say("  %-10s %s" % (pkg, getattr(m, "__version__", "?")))
    except Exception as e:
        say("  %-10s MISSING (%s)" % (pkg, e))
say()

t_all = time.time()
for s in STEPS:
    say("=" * 72)
    say("== %s" % s)
    say("=" * 72)
    t0 = time.time()
    p = subprocess.run([sys.executable, s], cwd=HERE, capture_output=True, text=True)
    say(p.stdout)
    if p.stderr.strip():
        say("[stderr]\n" + p.stderr)
    say("-- %s: exit %d, %.1f s" % (s, p.returncode, time.time() - t0))
    if p.returncode != 0:
        say("STOPPED: %s failed" % s)
        sys.exit(1)
say("\nALL DONE in %.1f s" % (time.time() - t_all))
log.close()
