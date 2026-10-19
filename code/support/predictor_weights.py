"""
Prepare inputs, compute predictor weights, and verify the results.
"""

import json
import os
import warnings
from typing import NamedTuple

import numpy as np
import cvxpy as cp
from scipy.optimize import linprog, minimize

from data_preparation import (
    CA_PREDICTORS,
    GER_PREDICTORS,
    build_california,
    build_germany,
)

TOL = 1e-9


class PredictorWeightInputs(NamedTuple):
    tag: str
    X1: np.ndarray
    X0: np.ndarray
    donors: list
    weights: np.ndarray
    predictor_names: list


def _prepare_case(tag, data, predictor_names, weights_path):
    with open(weights_path) as f:
        donor_weights = json.load(f)["donor_weights"]
    # Match the predictor-matrix columns; omitted donors have zero weight.
    weights = np.array([donor_weights.get(donor, 0.0) for donor in data["donors"]],
                       dtype=float)
    return PredictorWeightInputs(tag, data["X1"], data["X0"], data["donors"],
                                 weights, predictor_names)


def predictor_weight_cases(data_dir, output_dir):
    california = build_california(os.path.join(data_dir, "smoking_data.csv"))
    yield _prepare_case("california_headline", california, CA_PREDICTORS,
                        os.path.join(output_dir, "california_diagonal_donor_weights.json"))
    yield _prepare_case("california_leakage_free", california, CA_PREDICTORS,
                        os.path.join(output_dir, "california_diagonal_donor_weights_leakage_free.json"))

    germany = build_germany(os.path.join(data_dir, "german_reunification.csv"))
    yield _prepare_case("germany", germany, GER_PREDICTORS,
                        os.path.join(output_dir, "germany_diagonal_donor_weights.json"))


def _restricted_qp(Z1, Zc, X1=None, Xc=None, signs=None):
    """
    Least squares over simplex weights, optionally within one sign piece.
    """
    m = Zc.shape[1]

    # rescale the objective: 
    # SLSQP's ftol is absolute, badly scaled losses stall the solver
    scale = max(1.0, float(np.abs(Z1).max()))
    Z1 = Z1 / scale
    Zc = Zc / scale

    def f(b):
        rz = Z1 - Zc @ b
        return rz @ rz

    def g(b):
        return -2 * Zc.T @ (Z1 - Zc @ b)

    cons = [dict(type="eq", fun=lambda b: b.sum() - 1.0,
                 jac=lambda b: np.ones(m))]
    if signs is not None:
        S = np.diag(signs.astype(float))
        cons.append(dict(
            type="ineq",
            fun=lambda b: S @ (X1 - Xc @ b),
            jac=lambda b: -S @ Xc,
        ))
    def violation(b):
        v = abs(b.sum() - 1.0) + np.abs(np.minimum(b, 0)).sum()
        if signs is not None:
            v += np.abs(np.minimum(signs * (X1 - Xc @ b), 0)).sum()
        return v

    best = None
    starts = [np.full(m, 1.0 / m)]
    rng = np.random.default_rng(0)
    for _ in range(3):
        x = rng.random(m)
        starts.append(x / x.sum())
    for x0 in starts:
        res = minimize(f, x0, jac=g, bounds=[(0, 1)] * m,
                       constraints=cons, method="SLSQP",
                       options=dict(maxiter=1000, ftol=1e-12))
        # accept any feasible solutions, even if
        # SLSQP reports non-convergence on the (tight) ftol
        if violation(res.x) < 1e-7 and (best is None or res.fun < best.fun):
            best = res
    # second pass with Hessian; the problem is a
    # convex QP, this should be reliable
    from scipy.optimize import LinearConstraint
    tc_cons = [LinearConstraint(np.ones((1, m)), 1.0, 1.0)]
    if signs is not None:
        S = np.diag(signs.astype(float))
        tc_cons.append(LinearConstraint(S @ Xc, -np.inf, S @ X1))
    H = 2 * Zc.T @ Zc
    res = minimize(f, starts[0], jac=g, hess=lambda b: H,
                   bounds=[(0, 1)] * m, constraints=tc_cons,
                   method="trust-constr",
                   options=dict(maxiter=3000, gtol=1e-12, xtol=1e-14))
    if violation(res.x) < 1e-7 and (best is None or res.fun < best.fun):
        best = res
    if best is not None:
        best.fun = float(best.fun) * scale ** 2   # back to original units
    return best


def _donor_geometry(X1, X0, w):
    """
    Residual and donor partition.
    """
    J = X0.shape[1]
    r = X1 - X0 @ w
    active = [j for j in range(J) if w[j] > 1e-7]
    inactive = [j for j in range(J) if w[j] <= 1e-7]
    if not active:
        raise ValueError("Donor weights must contain a positive weight.")
    return r, active, inactive


def _build_pd_optimality_conditions(X0, r, active, inactive):
    """
    Build V, the trace and active-donor equalities, and inactive-donor products.

    Each product is (X_j - X_j0)' V r and must be nonpositive.
    """
    K = X0.shape[0]
    j0 = active[0]
    V = cp.Variable((K, K), PSD=True)
    eqs = [cp.trace(V) == 1]
    for j in active[1:]:
        d = X0[:, j] - X0[:, j0]
        eqs.append(d @ V @ r == 0)
    products = ((j, (X0[:, j] - X0[:, j0]) @ V @ r) for j in inactive)
    return V, eqs, products


def _reported_number(value, decimals):
    return float(value) if decimals is None else round(float(value), decimals)


def _check_predictor_solution(problem):
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError("Predictor-weight optimization failed: " + problem.status)
    if problem.value is None or not np.isfinite(problem.value):
        raise RuntimeError("Predictor-weight optimization returned no finite optimum.")


def compute_pd_logdet_and_ranges(X1, X0, w, names, *, decimals=4, ranges=True):
    """
    Compute the PD log-determinant maximizer and predictor-weight ranges.

    Use decimals=None for full precision and ranges=False to skip the ranges.
    """
    K = X0.shape[0]
    r, active, inactive = _donor_geometry(X1, X0, w)
    V, eqs, products = _build_pd_optimality_conditions(X0, r, active, inactive)
    cons = eqs + [product <= 0 for _, product in products]

    prob = cp.Problem(cp.Maximize(cp.log_det(V)), cons)
    prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-12,
               tol_gap_rel=1e-12, tol_feas=1e-12)
    _check_predictor_solution(prob)
    Vstar = V.value.copy()
    eigs = np.linalg.eigvalsh(Vstar)
    out = dict(status=prob.status,
               logdet=float(prob.value),
               min_eig=float(eigs.min()),
               diag={names[k]: _reported_number(Vstar[k, k], decimals)
                     for k in range(K)})
    if not ranges:
        return Vstar, out

    identified_ranges = {}
    inaccurate = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # cvxpy's 'inaccurate' warning is reported below instead
        for k in range(K):
            lo = cp.Problem(cp.Minimize(V[k, k]), cons)
            lo.solve(solver=cp.CLARABEL)
            _check_predictor_solution(lo)
            hi = cp.Problem(cp.Maximize(V[k, k]), cons)
            hi.solve(solver=cp.CLARABEL)
            _check_predictor_solution(hi)
            if "inaccurate" in (lo.status + hi.status):
                inaccurate.append(names[k])
            lower = float(lo.value)
            if decimals is not None and abs(lower) < 0.5 * 10 ** (-decimals):
                lower = abs(lower)  # Avoid displaying negative zero.
            identified_ranges[names[k]] = [
                _reported_number(lower, decimals),
                _reported_number(hi.value, decimals),
            ]
    if inaccurate:
        warnings.warn(
            "CLARABEL reported an inaccurate solution for the identified range of: "
            + ", ".join(map(str, inaccurate)),
            RuntimeWarning,
            stacklevel=2,
        )
    out["ranges"] = identified_ranges
    return Vstar, out


def verify(X1, X0, Vstar, w):
    """
    Inner QP at V*: sqrt(V*) transform, Euclidean projection.
    """
    lam, U = np.linalg.eigh(Vstar)
    lam = np.clip(lam, 0, None)
    sq = U @ np.diag(np.sqrt(lam)) @ U.T
    res = _restricted_qp(sq @ X1, sq @ X0)
    return float(np.abs(res.x - w).max())


def analytic_center(X1, X0, w, names, *, decimals=4):
    """
    Maximize log det V plus the log of each slack that can be positive.

    Slacks that are always zero are excluded. Use decimals=None for full precision.
    """
    K = X0.shape[0]
    r, active, inactive = _donor_geometry(X1, X0, w)
    j0 = active[0]
    V, eqs, products = _build_pd_optimality_conditions(X0, r, active, inactive)
    slack = {j: -product for j, product in products}
    ineqs = [slack[j] >= 0 for j in inactive]

    # Check whether all inactive slacks can be positive together.
    strict = []
    if inactive:
        t = cp.Variable()
        joint = cp.Problem(cp.Maximize(t),
                           eqs + [slack[j] >= t for j in inactive])
        joint.solve(solver=cp.CLARABEL, tol_gap_abs=1e-12,
                    tol_gap_rel=1e-12, tol_feas=1e-12)
        _check_predictor_solution(joint)
        if joint.value > TOL:
            strict = list(inactive)
        else:
            # A slack that is always zero cannot enter the log barrier.
            for j in inactive:
                pj = cp.Problem(cp.Maximize(slack[j]), eqs + ineqs)
                pj.solve(solver=cp.CLARABEL, tol_gap_abs=1e-12,
                         tol_gap_rel=1e-12, tol_feas=1e-12)
                _check_predictor_solution(pj)
                if pj.value > TOL:
                    strict.append(j)

    obj = cp.log_det(V)
    if strict:
        obj = obj + cp.sum(cp.hstack([cp.log(slack[j]) for j in strict]))
    prob = cp.Problem(cp.Maximize(obj), eqs + ineqs)
    prob.solve(solver=cp.CLARABEL)
    _check_predictor_solution(prob)
    Vc = V.value
    eigs = np.linalg.eigvalsh(Vc)
    slacks = {j: float(-((X0[:, j] - X0[:, j0]) @ Vc @ r))
              for j in inactive}
    return Vc, dict(
        status=prob.status,
        n_inactive=len(inactive),
        n_implicit_equalities=len(inactive) - len(strict),
        min_eig=float(eigs.min()),
        min_strict_slack=(min(slacks[j] for j in strict)
                          if strict else None),
        diag={names[k]: _reported_number(Vc[k, k], decimals) for k in range(K)},
    )


def diag_sets(X0, X1, w):
    r, active, inactive = _donor_geometry(X1, X0, w)
    j0 = active[0]
    A_eq = (X0[:, active[1:]] - X0[:, [j0]]).T * r
    S = (X0[:, [j0]] - X0[:, inactive]).T * r
    return r, inactive, A_eq, S


def lp_max(c, A_eq, S, K):
    """
    Maximize c'd over the diagonal predictor weights that rationalize w.
    """
    Aeq = np.vstack([A_eq, np.ones((1, K))])
    beq = np.zeros(Aeq.shape[0])
    beq[-1] = 1.0
    res = linprog(-c, A_ub=-S, b_ub=np.zeros(S.shape[0]), A_eq=Aeq, b_eq=beq,
                  bounds=(0, None), method="highs")
    if res.status != 0:
        raise RuntimeError("Diagonal predictor-weight LP failed: " + res.message)
    return -float(res.fun)


def diag_centers(X0, X1, w, names, *, decimals=4):
    """
    Log-det maximizer and analytic center of the diagonal rationalizers.

    Exclude weights and slacks that must be zero from the log barriers.
    Use decimals=None for full precision.
    """
    K = X0.shape[0]
    r, inactive, A_eq, S = diag_sets(X0, X1, w)
    support = [k for k in range(K) if lp_max(np.eye(K)[k], A_eq, S, K) > TOL]
    forced = [k for k in range(K) if k not in support]
    smax = [lp_max(S[m], A_eq, S, K) for m in range(len(inactive))]
    strict = [m for m in range(len(inactive)) if smax[m] > TOL]
    implicit = [m for m in range(len(inactive)) if smax[m] <= TOL]

    d = cp.Variable(K, nonneg=True)
    cons = [cp.sum(d) == 1]
    if len(A_eq):
        cons.append(A_eq @ d == 0)
    if len(S):
        cons.append(S @ d >= 0)
    cons += [d[k] == 0 for k in forced] + [S[m] @ d == 0 for m in implicit]
    out = {}
    objectives = {"logdet": cp.sum(cp.log(d[support])),
                  "analytic_center": cp.sum(cp.log(d[support])) + (cp.sum(cp.log(S[strict] @ d)) if strict else 0)}
    for label, obj in objectives.items():
        prob = cp.Problem(cp.Maximize(obj), cons)
        prob.solve(solver=cp.CLARABEL)
        _check_predictor_solution(prob)
        dv = np.asarray(d.value).ravel()
        out[label] = {names[k]: _reported_number(dv[k], decimals) for k in range(K)}
        out[label + "_status"] = prob.status
    out["support"] = [names[k] for k in support]
    out["forced_zero"] = [names[k] for k in forced]
    out["n_inactive_donors"] = len(inactive)
    out["n_implicit_equalities"] = len(implicit)
    out["residual_on_support"] = {names[k]: float(r[k]) for k in support}
    return out
