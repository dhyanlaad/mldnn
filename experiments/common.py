"""
common.py
=========
Shared utilities, C-accelerated fEM baseline runner, plotting, metric formatting,
and complete simulation data caching (.npz) for Experiments 1 through 5.
"""

from __future__ import annotations
import os
import ctypes
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gamma as sgamma
from pathlib import Path

import config

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.8,
    "savefig.dpi": 300
})

# Model type constants matching benchmark/fast_fem_all.c
MODEL_OU = 1
MODEL_GBM = 2
MODEL_LOGISTIC = 3
MODEL_NONLINEAR = 4
MODEL_CIR = 5
MODEL_TRIGONOMETRIC = 6

# Load C library
_lib_path = config.LIBFAST_FEM_PATH
if _lib_path.exists():
    _c_lib = ctypes.CDLL(str(_lib_path))
    _c_lib.solve_fem_generic_c.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int
    ]
else:
    _c_lib = None

def run_fast_fem(
    model_type: int,
    alpha: float,
    p1: float,
    p2: float,
    p3: float,
    y0: float,
    dB: np.ndarray,
    t_eval: np.ndarray,
    num_threads: int = config.NUM_WORKER_THREADS
) -> np.ndarray:
    """Run multi-threaded C-accelerated Fractional Euler-Maruyama solver on batch dB."""
    if _c_lib is None:
        raise RuntimeError(f"C extension not found at {_lib_path}. Compile it first.")
    
    if dB.ndim == 1:
        dB = dB[None, :]
    n_paths, n = dB.shape
    dt = 1.0 / n
    d = np.arange(1, n + 1, dtype=np.float64)
    wdet = (dt**alpha * (np.power(d, alpha) - np.power(d - 1.0, alpha)) / alpha) / sgamma(alpha)
    ksto = np.power(d * dt, alpha - 1.0) / sgamma(alpha)
    
    y_out = np.zeros((n_paths, len(t_eval)), dtype=np.float64)
    dB_c = np.ascontiguousarray(dB, dtype=np.float64)
    wdet_c = np.ascontiguousarray(wdet, dtype=np.float64)
    ksto_c = np.ascontiguousarray(ksto, dtype=np.float64)
    t_eval_c = np.ascontiguousarray(t_eval, dtype=np.float64)
    
    _c_lib.solve_fem_generic_c(
        n_paths, n, model_type, alpha, p1, p2, p3, y0,
        dB_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        wdet_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ksto_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        y_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(t_eval), t_eval_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), num_threads
    )
    return y_out

def save_experiment_cache(
    out_dir: Path,
    t_eval: np.ndarray,
    alphas: list[float],
    mhat_values: list[int],
    exact_dict: dict[float, np.ndarray],
    mldnn_dict: dict[float, dict[int, np.ndarray]],
    metrics_dict: dict[str, dict[int, dict[str, float]]],
    params: dict | None = None
):
    """
    Save complete simulation trajectories, errors, and metadata in a single .npz + .json
    so re-plotting and re-formatting can be done instantly without re-running simulations.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_file = out_dir / "data_cache.npz"
    meta_file = out_dir / "metadata.json"
    
    # Flatten arrays for npz storage
    save_arrays = {
        "t_eval": t_eval,
        "alphas": np.array(alphas, dtype=float),
        "mhat_values": np.array(mhat_values, dtype=int),
    }
    
    for a in alphas:
        save_arrays[f"exact_alpha_{a}"] = exact_dict[a]
        for m in mhat_values:
            save_arrays[f"mldnn_alpha_{a}_m_{m}"] = mldnn_dict[a][m]
            
    np.savez_compressed(cache_file, **save_arrays)
    
    meta = {
        "alphas": alphas,
        "mhat_values": mhat_values,
        "metrics": metrics_dict,
        "params": params or {},
    }
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)

def load_experiment_cache(out_dir: Path) -> dict:
    """Load cached simulation data for instant re-plotting."""
    cache_file = out_dir / "data_cache.npz"
    meta_file = out_dir / "metadata.json"
    
    if not cache_file.exists() or not meta_file.exists():
        raise FileNotFoundError(f"Cache files not found in {out_dir}")
        
    data = np.load(cache_file)
    with open(meta_file, "r") as f:
        meta = json.load(f)
        
    alphas = meta["alphas"]
    mhat_values = meta["mhat_values"]
    t_eval = data["t_eval"]
    
    exact_dict = {a: data[f"exact_alpha_{a}"] for a in alphas}
    mldnn_dict = {
        a: {m: data[f"mldnn_alpha_{a}_m_{m}"] for m in mhat_values}
        for a in alphas
    }
    
    return {
        "t_eval": t_eval,
        "alphas": alphas,
        "mhat_values": mhat_values,
        "exact_dict": exact_dict,
        "mldnn_dict": mldnn_dict,
        "metrics": meta["metrics"],
        "params": meta.get("params", {})
    }

def plot_qq(exact_terminal: np.ndarray, approx_terminal: np.ndarray, save_path: Path, title: str):
    """Generate Quantile-Quantile (QQ) plot at terminal time t = 1.0."""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    q_exact = np.sort(exact_terminal)
    q_approx = np.sort(approx_terminal)
    
    min_val = min(q_exact[0], q_approx[0])
    max_val = max(q_exact[-1], q_approx[-1])
    margin = (max_val - min_val) * 0.05
    
    ax.plot([min_val - margin, max_val + margin], [min_val - margin, max_val + margin],
            'r--', label="Reference Line ($y=x$)")
    
    ax.scatter(q_exact, q_approx, color='navy', alpha=0.6, s=15, label="Sample Quantiles")
    ax.set_xlabel("Benchmark Quantiles (Exact / fEM)")
    ax.set_ylabel("MLDNN Quantiles")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.set_xlim(min_val - margin, max_val + margin)
    ax.set_ylim(min_val - margin, max_val + margin)
    
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

def plot_paths_comparison(
    t_grid: np.ndarray,
    exact_paths: np.ndarray,
    approx_paths: np.ndarray,
    save_path: Path,
    title: str,
    n_sample_lines: int = 4
):
    """Plot mean trajectories with 95% confidence intervals and sample paths."""
    fig, ax = plt.subplots(figsize=(7, 5))
    
    mean_exact = np.mean(exact_paths, axis=0)
    std_exact = np.std(exact_paths, axis=0)
    
    mean_approx = np.mean(approx_paths, axis=0)
    std_approx = np.std(approx_paths, axis=0)
    
    ax.fill_between(t_grid, mean_approx - 1.96 * std_approx, mean_approx + 1.96 * std_approx,
                    color='gray', alpha=0.25, label="MLDNN 95% CI")
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i in range(min(n_sample_lines, len(exact_paths))):
        c = colors[i % len(colors)]
        ax.plot(t_grid, exact_paths[i], linestyle="--", alpha=0.5, color=c,
                label="Benchmark Paths" if i == 0 else "")
        ax.plot(t_grid, approx_paths[i], linestyle="-", alpha=0.9, color=c,
                label="MLDNN Paths" if i == 0 else "")
        
    ax.plot(t_grid, mean_exact, 'r:', linewidth=2.2, label="Benchmark Mean $\\mathbb{E}[X(t)]$")
    ax.plot(t_grid, mean_approx, 'k-', linewidth=2.0, label="MLDNN Mean $\\mathbb{E}[X(t)]$")
    
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("$X(t)$")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
