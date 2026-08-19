"""
exp3_gbm.py
===========
Fractional Geometric Brownian Motion (GBM) Experiment Suite:
Equation: D_t^alpha y(t) = mu * y(t) + sigma * y(t) * dW_t/dt, y(0) = 1.0
Parameters: mu = 0.3, sigma = 0.15, y0 = 1.0, R = 5000 paths, N = 65,536 steps
- alpha = 1.0: Exact geometric Ito benchmark
- alpha < 1.0: Fractional Euler-Maruyama (C fEM) benchmark
"""

from __future__ import annotations
import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from solver.parallel import solve_affine_fubini_batch
from solver.core_mldnn import brownian_paths, ml_vec
from experiments.common import run_fast_fem, MODEL_GBM

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "lines.linewidth": 1.8,
    "savefig.dpi": 300
})

def main():
    mu = 0.3
    sigma = 0.15
    y0 = 1.0
    n_paths = 5000
    n_steps = 65536
    t_eval = np.linspace(0.0, 1.0, 101)
    Nq = 64
    mhat_values = [2, 4, 8, 16, 24, 32]
    alphas_mean = [0.55, 0.65, 0.75, 0.85, 0.95, 1.00]
    
    out_dir = config.EXPORTS_DIR / "exp3_gbm" / "mu_03_sigma_015"
    out_dir.mkdir(parents=True, exist_ok=True)
    tex_fig_dir = ROOT / "tex" / "numerics" / "figures"
    tex_fig_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 85)
    print("RUNNING EXPERIMENTAL SUITE: GBM (mu = 0.3, sigma = 0.15)")
    print(f"Parameters: mu={mu}, sigma={sigma}, y0={y0}, R={n_paths:,} paths, N={n_steps:,} steps")
    print("=" * 85)
    
    # 1. Brownian Increments & Exact Ito Benchmark
    print(f"Generating Brownian increments ({n_paths:,} paths, {n_steps:,} steps)...")
    rng = np.random.default_rng(config.SEED)
    dB = brownian_paths(n_steps, n_paths, rng=rng, seed=config.SEED)
    
    print("Evaluating exact geometric Ito benchmark on continuous Brownian paths (alpha = 1.0)...")
    t_mesh = np.linspace(0.0, 1.0, n_steps + 1)
    W = np.zeros((n_paths, n_steps + 1), dtype=np.float64)
    W[:, 1:] = np.cumsum(dB, axis=1)
    
    fine_exact = y0 * np.exp((mu - 0.5 * (sigma ** 2)) * t_mesh[None, :] + sigma * W)
    exact_eval_a10 = np.array([np.interp(t_eval, t_mesh, fine_exact[p]) for p in range(n_paths)])
    
    # 2. Classical SDE MSE Convergence Sweep (alpha = 1.0)
    print("\n--- Running Classical SDE MSE Convergence Sweep (alpha = 1.0) ---")
    error_records = []
    
    for m in mhat_values:
        t0 = time.time()
        sol_m = solve_affine_fubini_batch(
            alpha=1.0, mhat=m, dB=dB, y0=y0, b0=0.0, b1=mu, s0=0.0, s1=sigma, Nq=max(Nq, m + 1), t_eval=t_eval, trace_order=1
        )
        elapsed = time.time() - t0
        
        diff = exact_eval_a10 - sol_m
        mse_t = np.mean(diff ** 2, axis=0)
        sup_mse = float(np.max(mse_t))
        l2_mse = float(np.trapz(mse_t, t_eval))
        
        error_records.append({
            "mhat": m,
            "Sup_t_MSE": sup_mse,
            "L2_MSE": l2_mse,
            "Sup_t_RMSE": np.sqrt(sup_mse),
            "L2_RMSE": np.sqrt(l2_mse),
            "Solve_Time_sec": elapsed
        })
        print(f"mhat = {m:2d} | Sup MSE: {sup_mse:.4e} | L2 MSE: {l2_mse:.4e} | Time: {elapsed:.2f}s")
        
    df_errors = pd.DataFrame(error_records)
    df_errors.to_csv(out_dir / "errors_table.csv", index=False)
    
    # 3. Multi-Alpha Mean Error Analysis (mhat = 32)
    print("\n--- Running Multi-Alpha Mean Error Analysis (mhat = 32) ---")
    mean_records = []
    
    for a in alphas_mean:
        print(f"Evaluating mean error for alpha = {a:.2f}...")
        exact_mean = y0 * ml_vec(a, 1.0, mu * (t_eval ** a))
        y_num = solve_affine_fubini_batch(
            alpha=a, mhat=32, dB=dB, y0=y0, b0=0.0, b1=mu, s0=0.0, s1=sigma, Nq=Nq, t_eval=t_eval, trace_order=1
        )
        num_mean = np.mean(y_num, axis=0)
        disc_l2 = float((1.0 / len(t_eval)) * np.sqrt(np.sum((num_mean - exact_mean) ** 2)))
        disc_linf = float(np.max(np.abs(num_mean - exact_mean)))
        mean_records.append({
            "alpha": a,
            "Discrete_L2": disc_l2,
            "Discrete_Linf": disc_linf
        })
        print(f"alpha = {a:.2f} | Discrete L2: {disc_l2:.4e} | Discrete Linf: {disc_linf:.4e}")
        
    df_mean = pd.DataFrame(mean_records)
    df_mean.to_csv(out_dir / "discrete_l2_mean_error_updated.csv", index=False)
    
    # 4. Terminal QQ Plots (alpha = 0.85 & alpha = 1.0, mhat = 32)
    print("\n--- Generating Terminal QQ Plots at t = 1.0 (mhat = 32) ---")
    exact_t1_a10 = exact_eval_a10[:, -1]
    exact_t1_a085 = run_fast_fem(MODEL_GBM, 0.85, 0.0, mu, sigma, y0, dB, np.array([1.0])).squeeze(-1)
    
    sol_t1_a10 = solve_affine_fubini_batch(
        alpha=1.0, mhat=32, dB=dB, y0=y0, b0=0.0, b1=mu, s0=0.0, s1=sigma, Nq=Nq, t_eval=np.array([1.0]), trace_order=1
    ).squeeze(-1)
    sol_t1_a085 = solve_affine_fubini_batch(
        alpha=0.85, mhat=32, dB=dB, y0=y0, b0=0.0, b1=mu, s0=0.0, s1=sigma, Nq=Nq, t_eval=np.array([1.0]), trace_order=1
    ).squeeze(-1)
    
    probs = np.linspace(0.005, 0.995, 200)
    fig_qq, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))
    axes = {0.85: ax1, 1.0: ax2}
    qq_data = {
        0.85: {"bench": exact_t1_a085, "sol": sol_t1_a085, "title": r"$\alpha = 0.85$"},
        1.0: {"bench": exact_t1_a10, "sol": sol_t1_a10, "title": r"$\alpha = 1.0$"}
    }
    
    for a, ax in axes.items():
        bench = qq_data[a]["bench"]
        sol = qq_data[a]["sol"]
        q_bench = np.quantile(bench, probs)
        q_sol = np.quantile(sol, probs)
        
        q_min = min(np.min(q_bench), np.min(q_sol))
        q_max = max(np.max(q_bench), np.max(q_sol))
        pad = 0.05 * (q_max - q_min)
        line_vals = np.linspace(q_min - pad, q_max + pad, 100)
        
        ax.plot(line_vals, line_vals, color='#7f7f7f', linestyle='--', linewidth=1.4, label='Reference ($y = x$)', zorder=1)
        ax.scatter(q_bench, q_sol, color='#1f77b4', alpha=0.7, s=18, edgecolors='none', label=r'MLDNN $\hat{m} = 32$', zorder=3)
        
        b_mean, b_std = np.mean(bench), np.std(bench)
        m_mean, m_std = np.mean(sol), np.std(sol)
        bench_lbl = "Exact" if np.isclose(a, 1.0) else "fEM"
        stats_text = (
            f"{bench_lbl}: $\\mu = {b_mean:.4f}$, $\\sigma = {b_std:.4f}$\n"
            f"MLDNN: $\\mu = {m_mean:.4f}$, $\\sigma = {m_std:.4f}$"
        )
        ax.text(
            0.05, 0.92, stats_text, transform=ax.transAxes,
            fontsize=9.5, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='gray')
        )
        
        ax.set_xlabel(f'{bench_lbl} Benchmark Quantiles ($t = 1.0$)', fontsize=10.5)
        ax.set_ylabel(r'MLDNN Quantiles ($t = 1.0$)', fontsize=10.5)
        ax.set_title(f'Fractional GBM: {qq_data[a]["title"]}', fontsize=11.5, fontweight='bold')
        ax.set_xlim(q_min - pad, q_max + pad)
        ax.set_ylim(q_min - pad, q_max + pad)
        ax.set_aspect('equal', 'box')
        ax.legend(frameon=True, loc='lower right', fontsize=9.5)
        
    fig_qq.tight_layout()
    qq_comb_path = tex_fig_dir / "qq_gbm_alpha085_alpha10_t1_combined.png"
    fig_qq.savefig(qq_comb_path, dpi=300)
    fig_qq.savefig(out_dir / "qq_gbm_alpha085_alpha10_t1_combined.png", dpi=300)
    plt.close(fig_qq)
    print(f">> QQ combined plot saved to {qq_comb_path}")

if __name__ == '__main__':
    main()

