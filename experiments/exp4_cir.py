"""
run_cir_mu03_sigma015_complete.py
=================================
Cox-Ingersoll-Ross (CIR) Process Experiment Suite (Corrected):
D_t^alpha y(t) = mu * y(t) + sigma * sqrt(y(t)) * dW_t/dt, y(0) = 1.0
Parameters: mu = 0.3, sigma = 0.15, y0 = 1.0, R = 5000 paths, N = 65,536 steps
"""

from __future__ import annotations
import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import scipy.special as sp
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from solver.core_mldnn import brownian_paths, basis_eval, Blocks
from solver.parallel import build_fubini_tensor
from experiments.common import run_fast_fem, MODEL_CIR

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "lines.linewidth": 1.8,
    "savefig.dpi": 300
})

def solve_cir_batched_torch(
    alpha: float,
    mhat: int,
    dB: np.ndarray,
    y0: float,
    mu: float,
    sigma: float,
    Nq: int,
    t_eval: np.ndarray,
    max_iter: int = 15,
    tol: float = 1e-8
) -> np.ndarray:
    n_paths, n_steps = dB.shape
    m1 = mhat + 1
    B = Blocks(alpha, mhat, None, Nq)
    
    # 1. Fubini tensor contraction
    M_tens, _ = build_fubini_tensor(alpha, mhat, n_steps)
    M_t = torch.from_numpy(M_tens)
    dB_t = torch.from_numpy(dB)
    Xi_t = torch.matmul(dB_t, M_t.T)
    
    omega_inv_t = (2.0 * alpha * torch.arange(m1, dtype=torch.float64) + 1.0)
    Xi_all = Xi_t.reshape(n_paths, m1, m1)
    S_all = Xi_all * omega_inv_t[None, None, :]
    
    PhiT_t = torch.from_numpy(B.PhiT)
    DetT_t = torch.from_numpy(B.DetT)
    # Correct StoT = PhiT @ S^T
    StoT_all = torch.matmul(PhiT_t.unsqueeze(0), S_all.transpose(1, 2))
    
    t_cheb_t = torch.from_numpy(B.t)
    if np.isclose(alpha, 1.0):
        c_alpha_t = torch.full_like(t_cheb_t, 0.5)
    else:
        c0 = sp.gamma(2.0 * alpha - 1.0) / (2.0 * (sp.gamma(alpha) ** 2))
        c_alpha_t = c0 * torch.pow(torch.clamp(t_cheb_t, min=0.0), 2.0 * alpha - 1.0)
        
    c_alpha_np = c_alpha_t.numpy()
    
    # Linearized initial guess about y0
    s0_0 = 0.5 * sigma * np.sqrt(y0)
    s1_0 = 0.5 * sigma / np.sqrt(y0)
    
    b0_eff = -0.25 * (sigma ** 2) * c_alpha_np
    b1_eff = np.full(Nq, mu)
    
    PhiT_batch = PhiT_t.unsqueeze(0).expand(n_paths, -1, -1)
    DetT_batch = DetT_t.unsqueeze(0).expand(n_paths, -1, -1)
    Zero_m1 = torch.zeros((n_paths, Nq, m1), dtype=torch.float64)
    
    J1 = torch.cat([PhiT_batch, -DetT_batch, -StoT_all], dim=2)
    R2_0 = torch.cat([-torch.from_numpy(b1_eff).unsqueeze(-1) * PhiT_batch, PhiT_batch, Zero_m1], dim=2)
    R3_0 = torch.cat([-s1_0 * PhiT_batch, Zero_m1, PhiT_batch], dim=2)
    Amat0 = torch.cat([J1, R2_0, R3_0], dim=1)
    
    rhs0 = torch.from_numpy(np.concatenate([np.full(Nq, y0), b0_eff, np.full(Nq, s0_0)])).unsqueeze(0).expand(n_paths, -1)
    
    AtA0 = torch.bmm(Amat0.transpose(1, 2), Amat0)
    reg_I = 1e-10 * torch.eye(3 * m1, dtype=torch.float64).unsqueeze(0)
    Atb0 = torch.bmm(Amat0.transpose(1, 2), rhs0.unsqueeze(-1))
    z = torch.linalg.solve(AtA0 + reg_I, Atb0).squeeze(-1)
    
    c_alpha_batch = c_alpha_t.unsqueeze(0)
    
    for it in range(max_iter):
        c = z[:, :m1]
        tb = z[:, m1:2*m1]
        ts = z[:, 2*m1:]
        
        N = torch.matmul(c, PhiT_t.T)
        N_clamped = torch.clamp(N, min=1e-8)
        sqrt_N = torch.sqrt(N_clamped)
        
        s_val = sigma * sqrt_N
        b_eff = mu * N - 0.25 * (sigma ** 2) * c_alpha_batch
        
        r1 = N - y0 - torch.matmul(tb, DetT_t.T) - torch.bmm(StoT_all, ts.unsqueeze(-1)).squeeze(-1)
        r2 = torch.matmul(tb, PhiT_t.T) - b_eff
        r3 = torch.matmul(ts, PhiT_t.T) - s_val
        
        F = torch.cat([r1, r2, r3], dim=1).unsqueeze(-1)
        norm_F = torch.max(torch.norm(F.squeeze(-1), dim=1)).item()
        if norm_F < tol:
            break
            
        sp_val = 0.5 * sigma / sqrt_N
        J2 = torch.cat([-mu * PhiT_batch, PhiT_batch, Zero_m1], dim=2)
        J3 = torch.cat([-sp_val.unsqueeze(-1) * PhiT_batch, Zero_m1, PhiT_batch], dim=2)
        J = torch.cat([J1, J2, J3], dim=1)
        
        JtJ = torch.bmm(J.transpose(1, 2), J)
        JtF = torch.bmm(J.transpose(1, 2), -F)
        dz = torch.linalg.solve(JtJ + reg_I, JtF).squeeze(-1)
        z = z + dz
        
    c_final = z[:, :m1].numpy()
    Phi_eval = basis_eval(alpha, mhat, t_eval)
    return c_final @ Phi_eval

def main():
    mu = 0.3
    sigma = 0.15
    y0 = 1.0
    n_paths = 5000
    n_steps = 65536
    t_eval = np.linspace(0.0, 1.0, 101)
    Nq = 64
    mhat_values = [1, 2, 4, 8, 16, 24, 32, 40]
    alphas_mean = [0.55, 0.65, 0.75, 0.85, 0.95, 1.00]
    
    out_dir = config.EXPORTS_DIR / "exp4_cir" / "mu_03_sigma_015"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = config.FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)
    tex_fig_dir = ROOT / "tex" / "numerics" / "figures"
    tex_fig_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 85)
    print("RUNNING EXPERIMENTAL SUITE: COX-INGERSOLL-ROSS (CIR) PROCESS")
    print(f"Parameters: mu={mu}, sigma={sigma}, y0={y0}, R={n_paths:,} paths, N={n_steps:,} steps")
    print("=" * 85)
    
    t0_all = time.time()
    rng = np.random.default_rng(config.SEED)
    dB = brownian_paths(n_steps, n_paths, rng=rng, seed=config.SEED)
    
    # 1. Fine-mesh fEM Benchmark for alpha = 1.0
    print("\n--- Computing Fine C fEM Benchmark (alpha = 1.0) ---")
    t0_fem = time.time()
    exact_eval_a10 = run_fast_fem(MODEL_CIR, 1.0, mu, 0.0, sigma, y0, dB, t_eval)
    print(f">> Exact benchmark computed in {time.time() - t0_fem:.2f}s")
    
    # 2. alpha = 1.0 Truncation Sweep
    print("\n--- Running CIR (alpha = 1.0) Truncation Sweep ---")
    error_records = []
    cache_dict = {
        "t_eval": t_eval,
        "alphas": np.array([1.0]),
        "mhat_values": np.array(mhat_values),
        "exact_alpha_1.0": exact_eval_a10
    }
    
    for m in mhat_values:
        t_start = time.time()
        sol = solve_cir_batched_torch(1.0, m, dB, y0, mu, sigma, max(Nq, m + 1), t_eval)
        elapsed = time.time() - t_start
        cache_dict[f"mldnn_alpha_1.0_m_{m}"] = sol
        
        diff = exact_eval_a10 - sol
        mse_t = np.mean(diff ** 2, axis=0)
        sup_mse = float(np.max(mse_t))
        l2_mse = float(np.mean(mse_t))
        sup_rmse = float(np.sqrt(sup_mse))
        l2_rmse = float(np.sqrt(l2_mse))
        
        error_records.append({
            "mhat": m,
            "Sup_MSE": sup_mse,
            "L2_MSE": l2_mse,
            "Sup_RMSE": sup_rmse,
            "L2_RMSE": l2_rmse,
            "Solve_Time_sec": elapsed
        })
        print(f"mhat = {m:2d} | Sup MSE: {sup_mse:.4e} | L2 MSE: {l2_mse:.4e} | Time: {elapsed:.2f}s")
        
    df_errors = pd.DataFrame(error_records)
    csv_path = out_dir / "errors_table.csv"
    df_errors.to_csv(csv_path, index=False)
    
    md_table = df_errors.to_markdown(index=False)
    with open(out_dir / "table_errors.md", "w") as f:
        f.write("# CIR Process (alpha = 1.0, mu = 0.3, sigma = 0.15)\n\n")
        f.write(md_table + "\n")
        
    # 3. Multi-Alpha Mean Error Analysis against Exact Mittag-Leffler Mean
    print("\n--- Running Multi-Alpha Mean Error Analysis (Mittag-Leffler Mean) ---")
    mhat_mean_sweep = [2, 4, 8, 16, 24, 32]
    disc_l2_dict = {a: {} for a in alphas_mean}
    disc_linf_dict = {a: {} for a in alphas_mean}
    
    for alpha in alphas_mean:
        print(f"Evaluating mean error for alpha = {alpha:.2f}...")
        # Exact Mittag-Leffler mean: E[y(t)] = y0 * E_alpha(mu * t^alpha)
        z = mu * np.power(t_eval, alpha)
        ml_mean = np.zeros_like(t_eval)
        for k in range(120):
            term = np.power(z, k) / sp.gamma(alpha * k + 1.0)
            ml_mean += term
            if np.max(np.abs(term)) < 1e-18:
                break
        exact_mean = y0 * ml_mean
        cache_dict[f"exact_mean_alpha_{alpha}"] = exact_mean
        
        for m in mhat_mean_sweep:
            sol_m = solve_cir_batched_torch(alpha, m, dB, y0, mu, sigma, max(Nq, m + 1), t_eval)
            num_mean = np.mean(sol_m, axis=0)
            if m == 24:
                cache_dict[f"num_mean_alpha_{alpha}_m24"] = num_mean
                
            disc_l2 = float((1.0 / len(t_eval)) * np.sqrt(np.sum((num_mean - exact_mean) ** 2)))
            disc_linf = float(np.max(np.abs(num_mean - exact_mean)))
            
            disc_l2_dict[alpha][m] = disc_l2
            disc_linf_dict[alpha][m] = disc_linf
            
    df_disc_l2 = pd.DataFrame(disc_l2_dict).T
    df_disc_l2.index.name = "alpha"
    df_disc_l2.to_csv(out_dir / "discrete_l2_mean_error.csv")
    
    df_disc_linf = pd.DataFrame(disc_linf_dict).T
    df_disc_linf.index.name = "alpha"
    df_disc_linf.to_csv(out_dir / "discrete_linf_mean_error.csv")
    
    m24_summary = []
    for a in alphas_mean:
        m24_summary.append({
            "alpha": a,
            "Discrete_L2 (m=24)": f"{disc_l2_dict[a][24]:.4e}",
            "Discrete_Linf (m=24)": f"{disc_linf_dict[a][24]:.4e}"
        })
    df_m24 = pd.DataFrame(m24_summary)
    with open(out_dir / "mean_errors_table.md", "w") as f:
        f.write("# CIR Process Mean Error Analysis (mu = 0.3, sigma = 0.15)\n\n")
        f.write("## Discrete Mean Errors for mhat = 24\n\n")
        f.write(df_m24.to_markdown(index=False) + "\n\n")
        f.write("## Full Discrete L2 Mean Error Grid\n\n")
        f.write(df_disc_l2.to_markdown() + "\n\n")
        f.write("## Full Discrete Linf Mean Error Grid\n\n")
        f.write(df_disc_linf.to_markdown() + "\n")
        
    cache_path = out_dir / "data_cache.npz"
    np.savez_compressed(cache_path, **cache_dict)
    print(f">> Full simulation data cache written to {cache_path}")
    
    # 4. Terminal QQ Plots (alpha = 0.85 & alpha = 1.0, mhat = 32)
    print("\n--- Generating Terminal QQ Plots at t = 1.0 (mhat = 32) ---")
    exact_t1_a10 = exact_eval_a10[:, -1]
    
    print("Computing C fEM benchmark for alpha = 0.85 at t = 1.0...")
    t0_fem85 = time.time()
    exact_t1_a085 = run_fast_fem(MODEL_CIR, 0.85, mu, 0.0, sigma, y0, dB, np.array([1.0])).squeeze(-1)
    print(f">> fEM alpha=0.85 evaluated in {time.time() - t0_fem85:.2f}s")
    
    sol_t1_a10 = solve_cir_batched_torch(1.0, 32, dB, y0, mu, sigma, 64, np.array([1.0])).squeeze(-1)
    sol_t1_a085 = solve_cir_batched_torch(0.85, 32, dB, y0, mu, sigma, 64, np.array([1.0])).squeeze(-1)
    
    qq_cache_file = out_dir / "qq_raw_cache.npz"
    np.savez_compressed(
        qq_cache_file,
        exact_t1_a10=exact_t1_a10,
        mldnn_t1_a10=sol_t1_a10,
        exact_t1_a085=exact_t1_a085,
        mldnn_t1_a085=sol_t1_a085
    )
    print(f">> QQ raw endpoint cache written to {qq_cache_file}")
    
    probs = np.linspace(0.005, 0.995, 200)
    fig_qq, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))
    axes = {0.85: ax1, 1.0: ax2}
    qq_data = {
        0.85: {"bench": exact_t1_a085, "sol": sol_t1_a085, "title": r"$\alpha = 0.85$"},
        1.0: {"bench": exact_t1_a10, "sol": sol_t1_a10, "title": r"$\alpha = 1.0$"}
    }
    
    for alpha, ax in axes.items():
        bench = qq_data[alpha]["bench"]
        sol = qq_data[alpha]["sol"]
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
        bench_lbl = "Milstein" if np.isclose(alpha, 1.0) else "fEM"
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
        ax.set_title(f'Fractional CIR: {qq_data[alpha]["title"]}', fontsize=11.5, fontweight='bold')
        ax.set_xlim(q_min - pad, q_max + pad)
        ax.set_ylim(q_min - pad, q_max + pad)
        ax.set_aspect('equal', 'box')
        ax.legend(frameon=True, loc='lower right', fontsize=9.5)
        
    fig_qq.tight_layout()
    qq_comb_path = tex_fig_dir / "qq_cir_alpha085_alpha10_t1_combined.png"
    fig_qq.savefig(qq_comb_path, dpi=300)
    fig_qq.savefig(out_dir / "qq_cir_alpha085_alpha10_t1_combined.png", dpi=300)
    plt.close(fig_qq)
    print(f">> QQ combined plot saved to {qq_comb_path}")
    
    # 5. Single Sample Path Realization Homotopy Limit Plot
    print("\n--- Generating Single Sample Path Homotopy Limit Plot (mhat = 32) ---")
    dB_single = dB[:1, :]
    alphas_homotopy = [0.75, 0.85, 0.95, 1.00]
    fem_paths = {}
    mldnn_paths = {}
    
    for a in alphas_homotopy:
        fem_paths[a] = run_fast_fem(MODEL_CIR, a, mu, 0.0, sigma, y0, dB_single, t_eval)[0]
        mldnn_paths[a] = solve_cir_batched_torch(a, 32, dB_single, y0, mu, sigma, 64, t_eval)[0]
        
    fig_path, (ax_fem, ax_mldnn) = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    colors = {0.75: '#1f77b4', 0.85: '#ff7f0e', 0.95: '#2ca02c', 1.00: '#d62728'}
    
    for a in alphas_homotopy:
        if a == 1.00:
            ax_fem.plot(t_eval, fem_paths[a], color=colors[a], linewidth=2.4, linestyle='-', label=r'$\alpha = 1.00$ (SDE Benchmark)', zorder=5)
            ax_mldnn.plot(t_eval, mldnn_paths[a], color=colors[a], linewidth=2.4, linestyle='-', label=r'$\alpha = 1.00$ (SDE Benchmark)', zorder=5)
        else:
            ax_fem.plot(t_eval, fem_paths[a], color=colors[a], linewidth=1.8, linestyle='-', label=rf'$\alpha = {a:.2f}$')
            ax_mldnn.plot(t_eval, mldnn_paths[a], color=colors[a], linewidth=1.8, linestyle='-', label=rf'$\alpha = {a:.2f}$')
            
    ax_fem.set_title(r'(a) High-Resolution fEM Benchmark ($N = 65{,}536$)', fontsize=12, fontweight='bold')
    ax_fem.set_xlabel(r'Time $t$', fontsize=11)
    ax_fem.set_ylabel(r'State $y(t)$', fontsize=11)
    ax_fem.legend(frameon=True, loc='best', fontsize=10)
    
    ax_mldnn.set_title(r'(b) MLDNN Spectral Neural Solution ($\hat{m} = 32$)', fontsize=12, fontweight='bold')
    ax_mldnn.set_xlabel(r'Time $t$', fontsize=11)
    ax_mldnn.legend(frameon=True, loc='best', fontsize=10)
    
    fig_path.tight_layout()
    sample_path_fig = tex_fig_dir / "cir_sample_path_alpha_convergence.png"
    fig_path.savefig(sample_path_fig, dpi=300)
    fig_path.savefig(figures_dir / "cir_sample_path_alpha_convergence.png", dpi=300)
    plt.close(fig_path)
    print(f">> Sample path figure saved to {sample_path_fig}")
    
    metadata = {
        "model": "Cox-Ingersoll-Ross (CIR) Process (Corrected)",
        "equation": "D_t^alpha y(t) = mu * y(t) + sigma * sqrt(y(t)) * dW_t/dt",
        "mu": mu,
        "sigma": sigma,
        "y0": y0,
        "n_paths": n_paths,
        "n_steps": n_steps,
        "total_runtime_sec": time.time() - t0_all
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
        
    print("\n" + "=" * 85)
    print(f"ALL CIR EXPERIMENTS COMPLETED IN {time.time() - t0_all:.2f}s")
    print("=" * 85)

if __name__ == '__main__':
    main()
