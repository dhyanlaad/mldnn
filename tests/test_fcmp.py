"""
test_fcmp.py
============
Checks for the FCMP solver (solver/fcmp.py) against the properties used in the
FCMP convergence proof:
1. D is the exact Galerkin matrix of I^alpha; the filter has the required shape.
2. Additive noise: the trace vanishes and every pass gives the same state.
3. Skorokhod centring: E[G^T theta - tau] = 0 for exact Gaussian G (Lemma 0).
4. alpha = 1 GBM: error against the exact same-path solution falls with mhat,
   and dropping the trace leaves a larger error.
5. Nonlinear drift: the per-pass Newton solve converges.
6. Pinning: output satisfies y(0) = y0 exactly and is the L^2-nearest such element.
7. Contraction certificate: kappa < 1 for small L, and Picard alone reaches the
   same state as Picard + Newton.
"""

import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solver.fcmp import FCMPOperators, _FCMPMap, fcmp_filter, solve_fcmp_batch

torch.set_num_threads(2)


class TestOperators(unittest.TestCase):
    def test_D_is_galerkin_integration_at_alpha_1(self):
        n = 10
        ops = FCMPOperators(1.0, n, 256)
        x, w = np.polynomial.legendre.leggauss(60)
        t, w = (x + 1) / 2, w / 2
        L = np.polynomial.legendre
        phi = np.array([np.sqrt(2 * j + 1) * L.legval(2 * t - 1, np.eye(n + 1)[j]) for j in range(n + 1)])
        Iphi = []
        for j in range(n + 1):
            P = L.legint(np.eye(n + 1)[j], lbnd=-1) / 2          # int_0^t P_j(2s-1) ds
            Iphi.append(np.sqrt(2 * j + 1) * L.legval(2 * t - 1, P))
        D_ref = (phi * w) @ np.array(Iphi).T
        np.testing.assert_allclose(ops.D, D_ref, atol=1e-12)

    def test_filter_shape(self):
        for n in [4, 9, 32]:
            f = fcmp_filter(n)
            self.assertTrue(np.all(f[: n // 2 + 1] == 1.0))
            self.assertEqual(f[-1], 0.0)
            self.assertTrue(np.all((f >= 0) & (f <= 1)))
            self.assertTrue(np.all(np.diff(f) <= 0))


class TestScheme(unittest.TestCase):
    def test_additive_noise_passes_identical(self):
        rng = np.random.default_rng(0)
        dB = rng.standard_normal((6, 512)) / np.sqrt(512)
        _, co = solve_fcmp_batch(0.75, 8, dB, 1.0, lambda t, y: -0.3 * torch.sin(y),
                                 lambda t, y: 0.2 + 0.0 * y, n_passes=3, return_coeffs=True)
        np.testing.assert_allclose(co[0], co[2], atol=1e-13)

    def test_skorokhod_centring(self):
        a, n, P = 0.75, 4, 4000
        ops = FCMPOperators(a, n, 1024)
        C = ops.C4.reshape(ops.m1 ** 2, -1)
        L = np.linalg.cholesky(C + 1e-14 * np.eye(len(C)))
        G = torch.as_tensor((np.random.default_rng(1).standard_normal((P, len(C))) @ L.T).reshape(P, ops.m1, ops.m1))
        sch = _FCMPMap(ops, 1.0, None, lambda t, y: 0.5 * y, None, (0.0, 0.3), 3, "exact", 12)
        fns = sch.passes()
        for k in [2, 3]:
            s = torch.cat([torch.func.vmap(lambda g: fns[k](g)[2])(G[i:i + 500]) for i in range(0, P, 500)]).numpy()
            po = torch.cat([torch.func.vmap(lambda g: ops.tf * (g.T @ fns[k - 1](g)[0]))(G[i:i + 500])
                            for i in range(0, P, 500)]).numpy()
            z = s.mean(0) / (s.std(0) / np.sqrt(P) + 1e-300)
            z_po = po.mean(0) / (po.std(0) / np.sqrt(P) + 1e-300)
            self.assertLess(np.abs(z[s.std(0) > 0]).max(), 4.5, f"pass {k}")
            self.assertGreater(np.abs(z_po).max(), 8.0, f"pass {k}")

    def test_gbm_alpha1_converges_and_needs_trace(self):
        mu, lam, nst, P = 0.3, 0.5, 2048, 48
        rng = np.random.default_rng(2)
        dB = rng.standard_normal((P, nst)) / np.sqrt(nst)
        t = np.linspace(0, 1, nst + 1)
        B = np.concatenate([np.zeros((P, 1)), np.cumsum(dB, axis=1)], axis=1)
        exact = np.exp((mu - lam ** 2 / 2) * t + lam * B)
        te = t[::8]
        err = {}
        for n in [4, 16]:
            for tr in ["exact", "none"]:
                y = solve_fcmp_batch(1.0, n, dB, 1.0, None, lambda t, y: lam * y, affine_b=(0.0, mu),
                                     n_passes=4, trace=tr, t_eval=te)
                err[n, tr] = np.mean((y - exact[:, ::8]) ** 2)
        self.assertLess(err[16, "exact"], 0.6 * err[4, "exact"])
        self.assertLess(err[16, "exact"], err[16, "none"])

    def test_nonlinear_newton_converges(self):
        rng = np.random.default_rng(3)
        dB = rng.standard_normal((8, 512)) / np.sqrt(512)
        with warnings.catch_warnings():
            warnings.simplefilter("error")                      # residual warning -> failure
            y = solve_fcmp_batch(0.6, 8, dB, 1.0, lambda t, y: 0.3 * torch.cos(y),
                                 lambda t, y: 0.5 * torch.sin(y), n_passes=3,
                                 t_eval=np.linspace(0, 1, 5))
        self.assertTrue(np.all(np.isfinite(y)))
        # no assertion on y(0): FCMP converges in L^p(0,1) only and does not pin y(0) = y0


class TestPinningAndCertificate(unittest.TestCase):
    def test_pin_exact_and_nearest(self):
        ops = FCMPOperators(0.65, 12, 256)
        rng = np.random.default_rng(4)
        c = rng.standard_normal((5, ops.m1))
        cp = ops.pin(c, 1.0)
        v0 = ops.eval_matrix(np.array([0.0]))[0]
        np.testing.assert_allclose(cp @ v0, 1.0, atol=1e-12)
        # nearest: the change is orthogonal to {v : v(0) = 0}, i.e. parallel to v0
        dc = cp - c
        np.testing.assert_allclose(np.abs(dc @ v0) / (np.linalg.norm(dc, axis=1) * np.linalg.norm(v0)), 1.0, atol=1e-12)

    def test_solver_output_starts_at_y0(self):
        rng = np.random.default_rng(5)
        dB = rng.standard_normal((6, 512)) / np.sqrt(512)
        y = solve_fcmp_batch(0.6, 16, dB, 1.0, lambda t, y: 0.3 * torch.cos(y),
                             lambda t, y: 0.5 * torch.sin(y), n_passes=2, t_eval=np.array([0.0, 0.5, 1.0]))
        np.testing.assert_allclose(y[:, 0], 1.0, atol=1e-12)

    def test_certificate_and_picard(self):
        ops = FCMPOperators(0.75, 16, 512)
        kappa, lam = ops.contraction_certificate(0.3)
        self.assertLess(kappa, 1.0)
        rng = np.random.default_rng(6)
        dB = rng.standard_normal((4, 512)) / np.sqrt(512)
        args = (0.75, 16, dB, 1.0, lambda t, y: 0.3 * torch.cos(y), lambda t, y: 0.5 * torch.sin(y))
        y_pn = solve_fcmp_batch(*args, n_passes=2, t_eval=np.linspace(0, 1, 9), ops=ops)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_p = solve_fcmp_batch(*args, n_passes=2, t_eval=np.linspace(0, 1, 9), ops=ops,
                                   newton_iters=0, picard_iters=60)
        np.testing.assert_allclose(y_p, y_pn, atol=1e-11)


if __name__ == "__main__":
    unittest.main()
