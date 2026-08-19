"""
run_ou_theta03_sigma015.py
==========================
Experiment: Stochastic Ornstein-Uhlenbeck (OU) Process (alpha = 1.0, nonzero diffusion)
Equation: dy(t) = theta * (mu - y(t)) dt + sigma * dW_t,  y(0) = 1.0
Parameters: theta = 0.3, mu = 0.0, sigma = 0.15, y0 = 1.0, alpha = 1.0
Sweeps: mhat in {2, 4, 8, 16, 24, 32, 40}
Paths: 5000 paths, fine mesh = 65,536 steps

Benchmark: Exact analytic Ito integral solution
y_exact(t) = y0 * exp(-theta * t) + sigma * exp(-theta * t) * int_0^t exp(theta * s) dW_s
"""

from __future__ import annotations
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from solver.parallel import solve_affine_fubini_batch
from solver.core_mldnn import brownian_paths
from experiments.common import save_experiment_cache

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.8,
    "savefig.dpi": 300
})

def run_experiment():
    print("=" * 85)
    print("STOCHASTIC ORNSTEIN-UHLENBECK EXPERIMENT (alpha = 1.0, theta = 0.3, sigma = 0.15)")
    print("Equation: dy(t) = 0.3 * (0.0 - y(t)) dt + 0.15 * dW_t, y(0) = 1.0")
    print("=" * 85)
    
    out_dir = config.EXPORTS_DIR / "exp2_stochastic_ou" / "theta_03_sigma_015"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = config.FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    alpha = 1.0
    theta = 0.3
    mu = 0.0
    sigma = 0.15
    y0 = 1.0
    Nq = 64
    n_paths = 5000
    n_steps = 65536
    mhat_values = [1, 2, 4, 8, 16, 24, 32, 40]
    t_eval = np.linspace(0.0, 1.0, 101)
    
    # 1. Generate Brownian Increments
    print(f"Generating Brownian paths ({n_paths:,} paths, {n_steps:,} steps)...")
    rng = np.random.default_rng(config.SEED)
    dB = brownian_paths(n_steps, n_paths, rng=rng, seed=config.SEED)
    
    # 2. Compute Exact Analytic Ito Benchmark
    print("Evaluating exact analytic Ito benchmark on continuous Brownian paths...")
    t0 = time.time()
    t_mesh = np.linspace(0.0, 1.0, n_steps + 1)
    # y(t) = y0 * exp(-theta * t) + sigma * exp(-theta * t) * sum_{s <= t} exp(theta * s) * dB_s
    exp_s = np.exp(theta * t_mesh[:-1]) * dB  # (n_paths, n_steps)
    cum_int = np.zeros((n_paths, n_steps + 1), dtype=np.float64)
    cum_int[:, 1:] = np.cumsum(exp_s, axis=1)
    
    fine_exact = (
        y0 * np.exp(-theta * t_mesh[None, :]) +
        mu * (1.0 - np.exp(-theta * t_mesh[None, :])) +
        sigma * np.exp(-theta * t_mesh[None, :]) * cum_int
    )
    # Interpolate to t_eval
    exact_eval = np.array([np.interp(t_eval, t_mesh, fine_exact[p]) for p in range(n_paths)])
    print(f"Exact benchmark computed in {time.time() - t0:.2f}s")
    
    # 3. Solve via MLDNN for each mhat
    mldnn_dict = {alpha: {}}
    exact_dict = {alpha: exact_eval}
    
    records = []
    
    for mhat in mhat_values:
        t0 = time.time()
        y_mldnn = solve_affine_fubini_batch(
            alpha=alpha,
            mhat=mhat,
            dB=dB,
            y0=y0,
            b0=mu * theta,
            b1=-theta,
            s0=sigma,
            s1=0.0,
            Nq=Nq,
            t_eval=t_eval,
            trace_order=1
        )
        t_el = time.time() - t0
        mldnn_dict[alpha][mhat] = y_mldnn
        
        diff = exact_eval - y_mldnn
        mse_t = np.mean(diff ** 2, axis=0)  # MSE along paths at each t in [0, 1]
        sup_mse = float(np.max(mse_t))
        l2_mse = float(np.trapz(mse_t, t_eval))
        sup_rmse = float(np.sqrt(sup_mse))
        l2_rmse = float(np.sqrt(l2_mse))
        term_mse = float(mse_t[-1])
        
        records.append({
            "mhat": mhat,
            "Sup_t_MSE": sup_mse,
            "L2_MSE": l2_mse,
            "Sup_t_RMSE": sup_rmse,
            "L2_RMSE": l2_rmse,
            "Terminal_MSE": term_mse,
            "Solve_Time_sec": round(t_el, 3)
        })
        print(f"mhat = {mhat:2d} | Solve Time: {t_el:5.3f}s | Sup_t MSE: {sup_mse:.4e} | L2 MSE: {l2_mse:.4e}")
        
    df_results = pd.DataFrame(records)
    
    # 4. Save CSV and Markdown Table
    df_results.to_csv(out_dir / "errors_table.csv", index=False)
    
    table_headers = ["\\hat{m}", "\\text{Sup}_t \\text{ MSE}", "L_2 \\text{ MSE}", "\\text{Sup}_t \\text{ RMSE}", "L_2 \\text{ RMSE}", "\\text{Time (s)}"]
    table_lines = [
        "# Stochastic Ornstein-Uhlenbeck Process Error Table (alpha = 1.0)",
        "",
        f"Equation: dy(t) = -{theta} y(t) dt + {sigma} dW_t, y(0) = {y0}, mu = {mu}",
        f"Paths: {n_paths:,}, Fine Mesh Steps: {n_steps:,}, Collocation Points: Nq = {Nq}",
        "",
        "| $\\hat{m}$ | $\\text{Sup}_t \\text{ MSE}$ | $L_2 \\text{ MSE}$ | $\\text{Sup}_t \\text{ RMSE}$ | $L_2 \\text{ RMSE}$ | Solve Time (s) |",
        "| :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    for r in records:
        table_lines.append(
            f"| {r['mhat']:2d} | {r['Sup_t_MSE']:.4e} | {r['L2_MSE']:.4e} | {r['Sup_t_RMSE']:.4e} | {r['L2_RMSE']:.4e} | {r['Solve_Time_sec']:.3f} |"
        )
    md_table = "\n".join(table_lines)
    with open(out_dir / "table_errors.md", "w") as f:
        f.write(md_table)
        f.write("\n")
        
    print("\n" + "=" * 85)
    print("FINAL SUMMARY TABLE:")
    print("=" * 85)
    print(df_results.to_markdown(index=False))
    
    # 4. Multi-Alpha Mean Error Analysis (mhat = 32)
    print("\n--- Running Multi-Alpha Mean Error Analysis (mhat = 32) ---")
    alphas_mean = [0.55, 0.65, 0.75, 0.85, 0.95, 1.00]
    mean_records = []
    
    for a in alphas_mean:
        print(f"Evaluating mean error for alpha = {a:.2f}...")
        # Exact Mittag-Leffler mean: E[y(t)] = y0 * E_alpha(-theta * t^alpha)
        exact_mean = y0 * ml_vec(a, 1.0, -theta * (t_eval ** a))
        y_num = solve_affine_fubini_batch(
            alpha=a,
            mhat=32,
            dB=dB,
            y0=y0,
            b0=mu * theta,
            b1=-theta,
            s0=sigma,
            s1=0.0,
            Nq=Nq,
            t_eval=t_eval,
            trace_order=1
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
    
    # 5. Generate Side-by-Side Quantile-Quantile (QQ) Plot at t = 1.0 (alpha = 0.85 & alpha = 1.0)
    print("\nGenerating Side-by-Side Quantile-Quantile (QQ) Plot at t = 1.0...")
    from experiments.common import run_fast_fem, MODEL_OU
    
    exact_t1_a10 = exact_eval[:, -1]
    exact_t1_a085 = run_fast_fem(MODEL_OU, 0.85, 0.0, -theta, sigma, y0, dB, np.array([1.0])).squeeze(-1)
    
    sol_t1_a10 = solve_affine_fubini_batch(
        alpha=1.0, mhat=32, dB=dB, y0=y0, b0=0.0, b1=-theta, s0=sigma, s1=0.0, Nq=Nq, t_eval=np.array([1.0]), trace_order=1
    ).squeeze(-1)
    sol_t1_a085 = solve_affine_fubini_batch(
        alpha=0.85, mhat=32, dB=dB, y0=y0, b0=0.0, b1=-theta, s0=sigma, s1=0.0, Nq=Nq, t_eval=np.array([1.0]), trace_order=1
    ).squeeze(-1)
    
    qq_cache_file = out_dir / "qq_raw_cache.npz"
    np.savez_compressed(
        qq_cache_file,
        exact_t1_a10=exact_t1_a10,
        mldnn_t1_a10=sol_t1_a10,
        exact_t1_a085=exact_t1_a085,
        mldnn_t1_a085=sol_t1_a085
    )
    
    probs = np.linspace(0.005, 0.995, 200)
    fig_qq, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))
    axes = {0.85: ax1, 1.0: ax2}
    qq_data = {
        0.85: {"bench": exact_t1_a085, "sol": sol_t1_a085, "title": r"$\alpha = 0.85$"},
        1.0: {"bench": exact_t1_a10, "sol": sol_t1_a10, "title": r"$\alpha = 1.0$"}
    }
    
    for a, ax in axes.items():
        bench_val = qq_data[a]["bench"]
        sol_val = qq_data[a]["sol"]
        
        q_bench = np.quantile(bench_val, probs)
        q_sol = np.quantile(sol_val, probs)
        
        q_min = min(np.min(q_bench), np.min(q_sol))
        q_max = max(np.max(q_bench), np.max(q_sol))
        pad = 0.05 * (q_max - q_min)
        line_range = np.linspace(q_min - pad, q_max + pad, 100)
        
        ax.plot(line_range, line_range, 'k--', linewidth=1.5, label='Reference line ($y = x$)')
        ax.scatter(q_bench, q_sol, color='#1f77b4', alpha=0.7, s=18, edgecolors='none', label=r'MLDNN $\hat{m} = 32$')
        
        m_bench, s_bench = np.mean(bench_val), np.std(bench_val)
        m_sol, s_sol = np.mean(sol_val), np.std(sol_val)
        
        bench_lbl = "Exact" if np.isclose(a, 1.0) else "fEM"
        stats_text = (
            f"{bench_lbl}: $\\mu = {m_bench:.4f},\\ \\sigma = {s_bench:.4f}$\n"
            f"MLDNN: $\\mu = {m_sol:.4f},\\ \\sigma = {s_sol:.4f}$"
        )
        ax.text(
            0.05, 0.92, stats_text, transform=ax.transAxes,
            fontsize=9.5, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='gray')
        )
        
        ax.set_xlabel(f'{bench_lbl} Benchmark Quantiles ($t = 1.0$)', fontsize=10.5)
        ax.set_ylabel(r'MLDNN Quantiles ($t = 1.0$)', fontsize=10.5)
        ax.set_title(f'Fractional OU: {qq_data[a]["title"]}', fontsize=11.5, fontweight='bold')
        ax.set_xlim(q_min - pad, q_max + pad)
        ax.set_ylim(q_min - pad, q_max + pad)
        ax.legend(frameon=True, loc='lower right', fontsize=9.5)
        ax.set_aspect('equal', 'box')
        
    fig_qq.tight_layout()
    tex_fig_path = ROOT / "tex" / "numerics" / "figures" / "qq_ou_alpha085_alpha10_t1_combined.png"
    fig_qq.savefig(tex_fig_path, dpi=300)
    fig_qq.savefig(out_dir / "qq_ou_alpha085_alpha10_t1_combined.png", dpi=300)
    plt.close(fig_qq)
    print(f">> Side-by-side QQ Plot saved to {tex_fig_path}")

if __name__ == '__main__':
    run_experiment()

