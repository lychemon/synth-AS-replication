"""Regression tests for donor-weight optimization and visible faces."""

import csv
import itertools
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "support"))

import simplex_solver
import visible_facet_solver as facets


class SimplexSolverTests(unittest.TestCase):
    def test_singular_affine_bound_does_not_discard_an_exact_fit(self):
        Z1 = np.array([0.0])
        Zc = np.array([[0.0, 1.0, 1.0]])
        self.assertLessEqual(simplex_solver.affine_bound(Z1, Zc), 1e-10)

    def test_bounds_stay_below_the_simplex_minimum(self):
        Z1 = np.array([2.0, 3.0])
        Zc = np.array([[0.0, 1.0], [0.0, 0.0]])
        # The nearest point is (1, 0), with squared distance 10.
        self.assertLessEqual(simplex_solver.affine_bound(Z1, Zc), 10.0)
        for weights in (np.array([0.5, 0.5]), np.array([1.0, 0.0])):
            bound = simplex_solver.simplex_lower_bound(Z1, Zc, weights)
            self.assertGreaterEqual(bound, 0.0)
            self.assertLessEqual(bound, 10.0)

    def test_simplex_exact_fit(self):
        Zc = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        loss, weights = simplex_solver.qp_simplex(np.array([0.25, 0.5]), Zc)
        self.assertLess(loss, 1e-12)
        np.testing.assert_allclose(weights, [0.25, 0.25, 0.5], atol=1e-6)

    def test_utah_placebo_subproblem(self):
        donors = ["California", "Colorado", "Kansas", "Ohio", "Pennsylvania", "Virginia"]
        path = Path(__file__).resolve().parents[1] / "data" / "smoking_data.csv"
        with path.open() as handle:
            rows = sorted(csv.DictReader(handle), key=lambda row: float(row["year"]))
        outcomes = {
            state: [float(row["cigsale"]) for row in rows
                    if row["state"] == state and 1970 <= float(row["year"]) <= 1988]
            for state in ["Utah"] + donors
        }
        # Check the CSV values and the precision used in the Stata panel.
        for dtype in (np.float64, np.float32):
            with self.subTest(precision=dtype.__name__):
                Z1 = np.asarray(outcomes["Utah"], dtype=dtype).astype(float)
                Zc = np.asarray([outcomes[name] for name in donors], dtype=dtype).T.astype(float)
                # The optimum lies on the California-Pennsylvania edge.
                direction = Zc[:, 0] - Zc[:, 4]
                share = float(direction @ (Z1 - Zc[:, 4]) / (direction @ direction))
                expected = np.array([share, 0, 0, 0, 1 - share, 0])
                residual = Z1 - Zc @ expected
                gradient = -2 * Zc.T @ residual
                self.assertGreaterEqual(float((gradient - gradient @ expected).min()), -1e-7)
                loss, weights = simplex_solver.qp_simplex(Z1, Zc)
                np.testing.assert_allclose(weights, expected, rtol=0, atol=1e-7)
                self.assertAlmostEqual(loss, float(residual @ residual), places=6)

    def test_extra_inequality_and_equality(self):
        loss, weights = simplex_solver.qp_simplex(
            np.array([0.0, 1.0, 0.0]), np.eye(3),
            G=np.array([[0.0, 1.0, 0.0]]), h=np.array([0.4]),
            A_extra=np.array([[0.0, 0.0, 1.0]]), b_extra=np.array([0.2]))
        np.testing.assert_allclose(weights, [0.4, 0.4, 0.2], atol=1e-6)
        self.assertAlmostEqual(loss, 0.56, places=6)

    def test_infeasible_constraints(self):
        result = simplex_solver.qp_simplex(
            np.array([0.0]), np.array([[0.0, 1.0]]),
            G=np.ones((1, 2)), h=np.array([0.5]))
        self.assertEqual(result, (None, None))

    def test_iteration_limit_is_not_reported_as_an_optimum(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with self.assertRaises(RuntimeError):
                simplex_solver.qp_simplex(
                    np.array([0.0, 1.0, 0.0]), np.eye(3), maxiter=1)


class VisibleFacetTests(unittest.TestCase):
    def assert_visible_facets(self, X0, X1, expected):
        found = facets.visible_facets(X0, X1)
        self.assertEqual(set(found), {frozenset(face) for face in expected})
        for normal, offset in found.values():
            self.assertTrue(facets.all_donors_in_halfspace(normal, offset, X0))
            self.assertTrue(facets.visible(normal, offset, X1))

    def test_triangle_ray_exits_at_a_vertex(self):
        X0 = np.array([[0.0, 1.0, 3.0], [0.0, 2.0, 1.0]])
        self.assert_visible_facets(X0, np.array([0.0, 5.0]), [{0, 1}, {1, 2}])

    def test_square_ray_exits_at_a_vertex(self):
        X0 = np.array([[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
        self.assert_visible_facets(X0, np.array([2.0, 2.0]), [{1, 2}, {2, 3}])

    def test_one_dimensional_hull(self):
        X0 = np.array([[0.0, 1.0, 2.0]])
        self.assert_visible_facets(X0, np.array([3.0]), [{2}])
        self.assert_visible_facets(X0, np.array([-1.0]), [{0}])

    def test_pd_search_keeps_a_zero_loss_facet(self):
        X0 = np.array([
            [-1.8795, 1.8844, 1.5958, -0.4763, -0.4228, -0.2198, -0.1197],
            [-0.8191, -0.4868, -0.8691, -1.0059, 1.5365, -0.9130, -0.0176],
            [-0.4572, -0.0578, -0.8537, -2.0094, -1.6474, 0.3899, 0.9028]])
        X1 = np.array([3.4503, -1.2623, -0.1721])
        Z0 = np.array([[10.0, 1.0, 10.0, 2.0, 0.0, 2.0, 1.0]])
        # Donor 4 is visible and gives zero loss. Singular bounds must retain it.
        loss, weights, _ = facets.solve_exterior_pd(X0, X1, np.zeros(1), Z0)
        self.assertLess(loss, 1e-9)
        self.assertLess(float(np.linalg.norm(Z0 @ weights) ** 2), 1e-9)


def normal_patterns_from_lp(face, X0):
    """Check the paper's weak-sign condition directly in donor coordinates."""
    cols = sorted(face)
    K, J = X0.shape
    base = X0[:, cols[0]]
    face_rows = (X0[:, cols[1:]] - base[:, None]).T
    donor_rows = np.array([X0[:, j] - base for j in range(J) if j not in face])
    accepted = set()
    for pattern in itertools.product((-1, 1), repeat=K):
        signs = np.array(pattern)
        # With matching signs, signs @ normal = 1 excludes the zero normal.
        result = linprog(
            np.zeros(K), A_ub=np.vstack([donor_rows, -np.diag(signs)]),
            b_ub=np.zeros(len(donor_rows) + K),
            A_eq=np.vstack([face_rows, signs]),
            b_eq=np.concatenate([np.zeros(len(face_rows)), [1.0]]),
            bounds=[(None, None)] * K, method="highs")
        if result.success:
            accepted.add(pattern)
    return accepted


class DiagonalSignTests(unittest.TestCase):
    def test_normal_patterns_match_the_paper_with_both_enumeration_methods(self):
        X0 = np.array([[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
        for face in ({2, 3}, {2}):
            expected = normal_patterns_from_lp(face, X0)
            for cap in (2000, 0):
                with self.subTest(face=face, cap=cap):
                    with patch.object(facets, "_RAY_COMB_CAP", cap):
                        patterns = facets.allowed_sign_patterns(face, X0)
                    self.assertEqual({tuple(s) for s in patterns}, expected)

    def test_zero_normal_coordinate_allows_either_sign(self):
        X0 = np.array([[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
        expected = {(-1, 1), (1, 1)}
        for normal in (None, np.array([0.0, 1.0])):
            patterns = facets.allowed_sign_patterns({2, 3}, X0, normal)
            self.assertEqual({tuple(s) for s in patterns}, expected)

    def test_zero_normal_does_not_allow_any_pattern(self):
        X0 = np.array([[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
        self.assertEqual(facets.allowed_sign_patterns({2, 3}, X0, np.zeros(2)), [])

    def test_closed_region_can_contain_only_a_boundary_vertex(self):
        X0 = np.array([[0.0, 1.0, 3.0], [0.0, 2.0, 1.0]])
        X1 = np.array([1.0, 3.0])
        patterns = facets.feasible_pieces({1, 2}, X0, X1, np.array([1.0, 2.0]))
        self.assertEqual({tuple(s) for s in patterns}, {(1, 1)})
        G, h = facets.piece_rows(np.array([1, 1]), X0[:, [1, 2]], X1)
        _, weights = simplex_solver.qp_simplex(np.array([0.0]),
                                              np.array([[1.0, 0.0]]), G, h)
        np.testing.assert_allclose(weights, [1.0, 0.0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
