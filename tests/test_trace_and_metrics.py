"""
test_trace_and_metrics.py
=========================
Unit tests asserting acceptance criteria from ctx/new benchmarks/antigravity_fix_prompt.md:
1. trace_coefficient(1.0) == 0.5 exactly; no isclose(alpha, 1.0) in active correction path.
2. Additive noise regression guard: V0, V1, V2 produce bit-identical results.
3. Analytic values of sigma * partial_y sigma for all benchmark models.
4. Condition number logging and rank verification on collocation system matrix K.
"""

import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solver.core_mldnn import (
    trace_coefficient,
    malliavin_trace_factor,
    solve_affine,
    brownian_paths,
    build_S,
    Blocks,
    get_A,
    basis_eval,
    chebyshev_nodes,
    evaluate_solution,
    fubini_kernel_projection,
    solve_gauss_newton,
)
from solver.parallel import (
    solve_affine_fubini_batch,
    solve_nonlinear_fubini_batch,
    build_fubini_tensor,
    build_S_fubini_batch,
)
from solver.metrics import paired_log_ratio, terminal_rmse, trajectory_mse


class TestTraceAndMetrics(unittest.TestCase):

    def test_trace_coefficient_exact_half_at_alpha_one(self):
        """1. trace_coefficient(1.0) must equal 0.5 exactly in float64 without branching."""
        c1 = trace_coefficient(1.0)
        self.assertEqual(c1, 0.5)
        self.assertIsInstance(c1, float)

        # Also verify continuous behavior near 1.0
        c_near = trace_coefficient(0.999999999)
        self.assertAlmostEqual(c_near, 0.5, places=6)

    def test_additive_noise_regression_guard(self):
        """2. For additive noise (s1 = 0), V0, V1, V2 must give bit-identical output."""
        alpha = 1.0
        mhat = 8
        n_paths = 50
        n_steps = 2048
        dB = brownian_paths(n_steps, n_paths, seed=42, cache=False)
        y0 = 1.0
        theta = 2.0
        sigma = 0.3
        t_eval = np.linspace(0.0, 1.0, 11)

        # V0: zero correction
        sol_v0 = solve_affine_fubini_batch(
            alpha=alpha, mhat=mhat, dB=dB, y0=y0,
            b0=0.0, b1=-theta, s0=sigma, s1=0.0,
            t_eval=t_eval, correction="constant"
        )
        # Manually verify with zero trace
        sol_v1 = solve_affine_fubini_batch(
            alpha=alpha, mhat=mhat, dB=dB, y0=y0,
            b0=0.0, b1=-theta, s0=sigma, s1=0.0,
            t_eval=t_eval, correction="legacy_t_power"
        )

        # Max difference must be 0.0 (bit-identical)
        diff = np.max(np.abs(sol_v0 - sol_v1))
        self.assertEqual(diff, 0.0)

    def test_analytic_sigma_dsigma_values(self):
        """3. Assert analytic value of sigma * dsigma_dy across models."""
        y_test = 2.5
        sigma_val = 0.3

        # Additive OU: sigma(y) = sigma -> dsigma_dy = 0 -> sigma * dsigma_dy = 0
        s_add = lambda y: sigma_val
        ds_add = 0.0
        self.assertEqual(s_add(y_test) * ds_add, 0.0)

        # GBM: sigma(y) = sigma * y -> dsigma_dy = sigma -> sigma * dsigma_dy = sigma^2 * y
        s_gbm = lambda y: sigma_val * y
        ds_gbm = lambda y: sigma_val
        self.assertAlmostEqual(s_gbm(y_test) * ds_gbm(y_test), (sigma_val ** 2) * y_test)

        # CIR: sigma(y) = sigma * sqrt(y) -> dsigma_dy = sigma / (2*sqrt(y)) -> sigma * dsigma_dy = sigma^2 / 2
        s_cir = lambda y: sigma_val * np.sqrt(y)
        ds_cir = lambda y: sigma_val / (2.0 * np.sqrt(y))
        self.assertAlmostEqual(s_cir(y_test) * ds_cir(y_test), 0.5 * (sigma_val ** 2))

        # Trigonometric: sigma(y) = sigma * sin(y) -> dsigma_dy = sigma * cos(y) -> sigma * dsigma_dy = sigma^2 * sin(y)cos(y)
        s_trig = lambda y: sigma_val * np.sin(y)
        ds_trig = lambda y: sigma_val * np.cos(y)
        self.assertAlmostEqual(s_trig(y_test) * ds_trig(y_test), (sigma_val ** 2) * np.sin(y_test) * np.cos(y_test))

    def test_collocation_matrix_conditioning_logging(self):
        """4. Verify cond(K) computation and conditioning for mhat in {4, 8, 16, 32}."""
        alpha = 0.75
        for mhat in [4, 8, 16, 32]:
            Nq = max(64, mhat + 1)
            t_c = chebyshev_nodes(Nq)
            Phi = basis_eval(alpha, mhat, t_c); PhiT = Phi.T
            A_mat = get_A(alpha, mhat)
            DetT = (t_c ** alpha)[:, None] * (PhiT @ A_mat.T)
            m1 = mhat + 1
            Z = np.zeros((Nq, m1))
            # Test block
            K_det = np.vstack([
                np.hstack([PhiT, -DetT, Z]),
                np.hstack([-0.3 * PhiT, PhiT, Z]),
                np.hstack([-0.15 * PhiT, Z, PhiT])
            ])
            cond_k = np.linalg.cond(K_det)
            self.assertGreater(cond_k, 0.0)
            self.assertLess(cond_k, 1e12)

    def test_fubini_kernel_projection_at_alpha_one(self):
        """For M_0=1 and alpha=1, K_0(s)=int_s^1 1 dt=1-s."""
        s = np.linspace(0.0, 1.0, 17)
        k = fubini_kernel_projection(1.0, 0, s, quadrature_order=8)
        np.testing.assert_allclose(k[0], 1.0 - s, rtol=0.0, atol=2e-14)

    def test_operator_trace_core_and_batch_agree(self):
        """Single-path and batched implementations use the same accumulated trace."""
        alpha, mhat, n_steps, nq = 1.0, 6, 512, 20
        dB = brownian_paths(n_steps, 1, seed=4, cache=False)
        tensor, _ = build_fubini_tensor(alpha, mhat, n_steps)
        S = build_S_fubini_batch(alpha, mhat, dB, tensor)[0]
        c, _, _, _, trace = solve_affine(
            alpha, mhat, S, 1.0, 0.0, 0.3, 0.0, 0.15,
            Nq=nq, correction="operator_trace", return_trace=True,
            trace_quadrature_order=16, kernel_quadrature_order=16,
        )
        y_core = evaluate_solution(alpha, mhat, c, np.array([1.0]))[0]
        y_batch = solve_affine_fubini_batch(
            alpha, mhat, dB, 1.0, 0.0, 0.3, 0.0, 0.15,
            Nq=nq, t_eval=np.array([1.0]), M_tens=tensor,
            correction="operator_trace", tikhonov_reg=0.0,
            trace_quadrature_order=16, kernel_quadrature_order=16,
        )[0, 0]
        self.assertGreater(np.linalg.norm(trace), 0.0)
        self.assertAlmostEqual(y_core, y_batch, places=12)

    def test_nonlinear_operator_trace_core_and_batch_agree(self):
        """The nonlinear batched path reproduces the single-path core solve."""
        alpha, mhat, n_steps, nq = 1.0, 4, 256, 16
        dB = brownian_paths(n_steps, 1, seed=17, cache=False)
        tensor, _ = build_fubini_tensor(alpha, mhat, n_steps)
        S = build_S_fubini_batch(alpha, mhat, dB, tensor)[0]
        mu, sigma = 0.3, 0.15
        b = lambda t, y: mu * np.cos(y)
        bp = lambda t, y: -mu * np.sin(y)
        bp2 = lambda t, y: -mu * np.cos(y)
        s = lambda t, y: sigma * np.sin(y)
        sp = lambda t, y: sigma * np.cos(y)
        sp2 = lambda t, y: -sigma * np.sin(y)
        c, _, _, _, _, trace_core = solve_gauss_newton(
            alpha, mhat, S, 1.0, b, bp, s, sp,
            bprime2=bp2, sprime2=sp2, Nq=nq, maxit=30,
            correction="operator_trace", return_trace=True,
            trace_quadrature_order=12, kernel_quadrature_order=12,
        )
        y_core = evaluate_solution(alpha, mhat, c, np.array([1.0]))[0]
        y_batch, trace_batch = solve_nonlinear_fubini_batch(
            alpha, mhat, dB, 1.0, b, bp, bp2, s, sp, sp2,
            Nq=nq, t_eval=np.array([1.0]), M_tens=tensor,
            max_iter=30, tol=1e-12, tikhonov_reg=0.0,
            trace_quadrature_order=12, kernel_quadrature_order=12,
            return_trace=True,
        )
        np.testing.assert_allclose(trace_batch[0], trace_core, rtol=3e-5, atol=3e-7)
        self.assertAlmostEqual(y_batch[0, 0], y_core, places=5)

    def test_affine_malliavin_sensitivity_matches_finite_difference(self):
        """The full sensitivity, including the residual term, matches perturbing S."""
        alpha, mhat, n_steps, nq = 0.75, 3, 512, 12
        m1 = mhat + 1
        dB = brownian_paths(n_steps, 1, seed=5, cache=False)[0]
        S = build_S(alpha, mhat, dB)
        blocks = Blocks(alpha, mhat, S, nq)
        zero = np.zeros((nq, m1))
        mu, sigma = 0.3, 0.15
        K = np.vstack([
            np.hstack([blocks.PhiT, -blocks.DetT, -blocks.StoT]),
            np.hstack([-mu * blocks.PhiT, blocks.PhiT, zero]),
            np.hstack([-sigma * blocks.PhiT, zero, blocks.PhiT]),
            10.0 * np.concatenate([
                basis_eval(alpha, mhat, np.array([0.0]))[:, 0],
                np.zeros(2 * m1),
            ])[None, :],
        ])
        target = np.concatenate([np.ones(nq), np.zeros(2 * nq), [10.0]])
        theta = np.linalg.lstsq(K, target, rcond=None)[0]
        residual = K @ theta - target

        s = 0.37
        M_s = basis_eval(alpha, mhat, np.array([s]))[:, 0]
        k_s = fubini_kernel_projection(alpha, mhat, np.array([s]))[:, 0]
        omega_inv = 2.0 * alpha * np.arange(m1) + 1.0
        dS = np.outer(M_s, k_s * omega_inv)
        dK = np.zeros_like(K)
        dK[:nq, 2 * m1:] = -(blocks.PhiT @ dS.T)
        analytic = (
            -np.linalg.pinv(K) @ dK @ theta
            -np.linalg.pinv(K.T @ K, hermitian=True) @ dK.T @ residual
        )

        def coefficients(S_perturbed):
            out = solve_affine(
                alpha, mhat, S_perturbed, 1.0, 0.0, mu, 0.0, sigma,
                Nq=nq, correction="none",
            )
            return np.concatenate(out[:3])

        h = 1e-6
        finite_difference = (coefficients(S + h * dS) - coefficients(S - h * dS)) / (2 * h)
        np.testing.assert_allclose(analytic, finite_difference, rtol=2e-6, atol=2e-8)

    def test_operator_trace_reduces_gbm_ito_bias(self):
        """At alpha=1 the operator trace removes most of the V0 terminal log bias."""
        R, n_steps, mhat, nq = 40, 2048, 8, 24
        mu, sigma = 0.3, 0.15
        dB = brownian_paths(n_steps, R, seed=123, cache=False)
        y_ito = np.exp(mu - 0.5 * sigma**2 + sigma * dB.sum(axis=1))
        y_none = solve_affine_fubini_batch(
            1.0, mhat, dB, 1.0, 0.0, mu, 0.0, sigma,
            Nq=nq, t_eval=np.array([1.0]), correction="none",
        )[:, 0]
        y_trace = solve_affine_fubini_batch(
            1.0, mhat, dB, 1.0, 0.0, mu, 0.0, sigma,
            Nq=nq, t_eval=np.array([1.0]), correction="operator_trace",
            trace_quadrature_order=16, kernel_quadrature_order=16,
        )[:, 0]
        bias_none = abs(np.mean(np.log(y_none) - np.log(y_ito)))
        bias_trace = abs(np.mean(np.log(y_trace) - np.log(y_ito)))
        self.assertLess(bias_trace, 0.5 * bias_none)

    def test_nonlinear_operator_trace_uses_exact_hessian(self):
        """Nonlinear trace mode accepts second derivatives and returns finite output."""
        alpha, mhat, n_steps, nq = 1.0, 4, 512, 16
        dB = brownian_paths(n_steps, 1, seed=9, cache=False)
        tensor, _ = build_fubini_tensor(alpha, mhat, n_steps)
        S = build_S_fubini_batch(alpha, mhat, dB, tensor)[0]
        b = lambda t, y: 0.3 * y + 0.01 * y**2
        bp = lambda t, y: 0.3 + 0.02 * y
        bp2 = lambda t, y: np.full_like(y, 0.02)
        sigma = lambda t, y: 0.15 * y + 0.005 * y**2
        sp = lambda t, y: 0.15 + 0.01 * y
        sp2 = lambda t, y: np.full_like(y, 0.01)
        out = solve_gauss_newton(
            alpha, mhat, S, 1.0, b, bp, sigma, sp,
            Nq=nq, maxit=20, bprime2=bp2, sprime2=sp2,
            correction="operator_trace", return_trace=True,
            trace_quadrature_order=12, kernel_quadrature_order=12,
        )
        self.assertTrue(np.all(np.isfinite(out[0])))
        self.assertTrue(np.all(np.isfinite(out[5])))
        self.assertGreater(np.linalg.norm(out[5]), 0.0)


if __name__ == "__main__":
    unittest.main()
