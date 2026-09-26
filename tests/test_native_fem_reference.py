"""Independent small-grid checks for the C fractional Euler reference."""

from __future__ import annotations

import ctypes
import platform
import unittest
from pathlib import Path

import numpy as np
from scipy.special import gamma

MODEL_OU = 1
MODEL_GBM = 2
MODEL_TRIGONOMETRIC = 6
_suffix = ".dylib" if platform.system() == "Darwin" else ".so"
_lib_path = Path(__file__).resolve().parent.parent / "benchmark" / f"libfast_fem{_suffix}"


def _native(model, alpha, p1, p2, p3, y0, increments, times, threads=2):
    n_paths, n = increments.shape
    dt = 1.0 / n
    lag = np.arange(1, n + 1, dtype=float)
    drift_weights = dt**alpha * (lag**alpha - (lag - 1)**alpha) / gamma(alpha + 1)
    noise_weights = (lag * dt) ** (alpha - 1) / gamma(alpha)
    output = np.zeros((n_paths, len(times)))
    lib = ctypes.CDLL(str(_lib_path))
    function = lib.solve_fem_generic_c
    pointer = ctypes.POINTER(ctypes.c_double)
    function.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        pointer, pointer, pointer, pointer, ctypes.c_int, pointer, ctypes.c_int,
    ]
    function.restype = None
    function(
        n_paths, n, model, alpha, p1, p2, p3, y0,
        increments.ctypes.data_as(pointer), drift_weights.ctypes.data_as(pointer),
        noise_weights.ctypes.data_as(pointer), output.ctypes.data_as(pointer),
        len(times), times.ctypes.data_as(pointer), threads,
    )
    return output


def _reference(model, alpha, p1, p2, p3, y0, increments, times):
    """Direct scalar implementation of the same discretized Volterra equation."""
    n_paths, n = increments.shape
    dt = 1.0 / n
    lag = np.arange(1, n + 1, dtype=float)
    drift_weights = dt**alpha * (lag**alpha - (lag - 1)**alpha) / gamma(alpha + 1)
    noise_weights = (lag * dt) ** (alpha - 1) / gamma(alpha)
    paths = np.empty((n_paths, n + 1))
    paths[:, 0] = y0
    for path in range(n_paths):
        drift = np.empty(n)
        noise = np.empty(n)
        for k in range(n):
            y = paths[path, k]
            if model == MODEL_OU:
                drift[k], diffusion = p1 * (p2 - y), p3
            elif model == MODEL_GBM:
                drift[k], diffusion = p1 * y, p3 * y
            elif model == MODEL_TRIGONOMETRIC:
                drift[k], diffusion = p1 * np.cos(y), p3 * np.sin(y)
            else:
                raise ValueError(model)
            noise[k] = diffusion * increments[path, k]
            paths[path, k + 1] = y0 + np.dot(drift[: k + 1], drift_weights[k::-1]) + np.dot(
                noise[: k + 1], noise_weights[k::-1]
            )
    mesh = np.arange(n + 1) * dt
    return np.vstack([np.interp(times, mesh, path) for path in paths])


@unittest.skipUnless(_lib_path.exists(), "build benchmark/libfast_fem first")
class NativeFEMReferenceTests(unittest.TestCase):
    def test_native_matches_independent_discretization(self):
        rng = np.random.default_rng(109)
        increments = rng.normal(0.0, 1 / np.sqrt(32), size=(3, 32))
        times = np.array([0.0, 0.37, 1.0])
        cases = (
            (MODEL_OU, 0.85, 0.3, 0.0, 0.15),
            (MODEL_GBM, 0.85, 0.3, 0.0, 0.15),
            (MODEL_TRIGONOMETRIC, 0.85, 0.3, 0.0, 0.15),
            (MODEL_OU, 1.0, 0.3, 0.0, 0.15),
        )
        for model, alpha, p1, p2, p3 in cases:
            with self.subTest(model=model, alpha=alpha):
                expected = _reference(model, alpha, p1, p2, p3, 1.0, increments, times)
                actual = _native(model, alpha, p1, p2, p3, 1.0, increments, times, 2)
                np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)

    def test_corrected_ou_and_gbm_drift_parameters(self):
        increments = np.zeros((1, 1024))
        times = np.array([1.0])
        ou = _native(MODEL_OU, 1.0, 0.3, 0.0, 0.0, 1.0, increments, times)[0, 0]
        gbm = _native(MODEL_GBM, 1.0, 0.3, 0.0, 0.0, 1.0, increments, times)[0, 0]
        self.assertAlmostEqual(ou, np.exp(-0.3), delta=5e-5)
        self.assertAlmostEqual(gbm, np.exp(0.3), delta=6e-5)


if __name__ == "__main__":
    unittest.main()
