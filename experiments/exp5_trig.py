"""
exp5_trig.py
============
Nonlinear Trigonometric SDE Experiment Suite:
Equation: D_t^alpha y(t) = mu * cos(y(t)) + sigma * sin(y(t)) * dW_t/dt, y(0) = 1.0
Parameters: mu = 0.3, sigma = 0.15, y0 = 1.0, R = 5000 paths, N = 65,536 steps
- alpha = 1.0: Milstein scheme benchmark
- alpha < 1.0: Fractional Euler-Maruyama (C fEM) benchmark
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
from experiments.common import run_fast_fem, MODEL_TRIGONOMETRIC

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "lines.linewidth": 1.8,
    "savefig.dpi": 300
})

def solve_trig_batched_torch(
    alpha: float,
    mhat: int,
    dB: np.ndarray,
    y0: float,
    mu: float,
    sigma: float,
    Nq: int,
    t_eval: np.ndarray,
    max_iter: int = 20,
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
    StoT_all = torch.matmul(PhiT_t.unsqueeze(0), S_all.transpose(1, 2))
    
    t_cheb_t = torch.from_numpy(B.t)
    if np.isclose(alpha, 1.0):
        c_alpha_t = torch.full_like(t_cheb_t, 0.5)
    else:
        c0 = sp.gamma(2.0 * alpha - 1.0) / (2.0 * (sp.gamma(alpha) ** 2))
        c_alpha_t = c0 * torch.pow(torch.clamp(t_cheb_t, min=0.0), 2.0 * alpha - 1.0)
        
    c_alpha_np = c_alpha_t.numpy()
    
    # Linearized initial guess about y0
    s0_0 = sigma * (np.sin(y0) - np.cos(y0) * y0)
    s1_0 = sigma * np.cos(y0)
    bp_0 = -mu * np.sin(y0) - c_alpha_np * (sigma ** 2) * np.cos(2.0 * y0)
    b_val_0 = mu * np.cos(y0) - c_alpha_np * (sigma ** 2) * np.sin(y0) * np.cos(y0)
    b0_eff = b_val_0 - bp_0 * y0
    b1_eff = bp_0
    
    PhiT_batch = PhiT_t.unsqueeze(0).expand(n_paths, -1, -1)
    DetT_batch = DetT_t.unsqueeze(0).expand(n_paths, -1, -1)
    Zero_m1 = torch.zeros((n_paths, Nq, m1), dtype=torch.float64)
    
    Phi_0 = torch.from_numpy(basis_eval(alpha, mhat, np.array([0.0]))).squeeze(-1)
    bc_weight = 10.0
    bc_row_c = (bc_weight * Phi_0).unsqueeze(0).unsqueeze(0).expand(n_paths, 1, -1)
    bc_row = torch.cat([bc_row_c, torch.zeros((n_paths, 1, 2 * m1), dtype=torch.float64)], dim=2)
    bc_rhs = (bc_weight * torch.tensor([y0], dtype=torch.float64)).unsqueeze(0).expand(n_paths, 1)
    
    J1 = torch.cat([PhiT_batch, -DetT_batch, -StoT_all], dim=2)
    R2_0 = torch.cat([-torch.from_numpy(b1_eff).unsqueeze(-1) * PhiT_batch, PhiT_batch, Zero_m1], dim=2)
    R3_0 = torch.cat([-s1_0 * PhiT_batch, Zero_m1, PhiT_batch], dim=2)
    Amat0 = torch.cat([J1, R2_0, R3_0, bc_row], dim=1)
    
    rhs0 = torch.cat([
        torch.from_numpy(np.concatenate([np.full(Nq, y0), b0_eff, np.full(Nq, s0_0)])).unsqueeze(0).expand(n_paths, -1),
        bc_rhs
    ], dim=1)
    
    AtA0 = torch.bmm(Amat0.transpose(1, 2), Amat0)
    reg_I = 1e-8 * torch.eye(3 * m1, dtype=torch.float64).unsqueeze(0)
    Atb0 = torch.bmm(Amat0.transpose(1, 2), rhs0.unsqueeze(-1))
    z = torch.linalg.solve(AtA0 + reg_I, Atb0).squeeze(-1)
    
    c_alpha_batch = c_alpha_t.unsqueeze(0)
    
    for it in range(max_iter):
        c = z[:, :m1]
        tb = z[:, m1:2 * m1]
        ts = z[:, 2 * m1:]
        
        N = torch.matmul(c, PhiT_t.T)
        sin_N = torch.sin(N)
        cos_N = torch.cos(N)
        
        s_val = sigma * sin_N
        b_eff = mu * cos_N - c_alpha_batch * (sigma ** 2) * (sin_N * cos_N)
        
        r1 = N - y0 - torch.matmul(tb, DetT_t.T) - torch.bmm(StoT_all, ts.unsqueeze(-1)).squeeze(-1)
        r2 = torch.matmul(tb, PhiT_t.T) - b_eff
        r3 = torch.matmul(ts, PhiT_t.T) - s_val
        r_bc = (torch.matmul(c, Phi_0) - y0).unsqueeze(-1) * bc_weight
        
        F = torch.cat([r1, r2, r3, r_bc], dim=1).unsqueeze(-1)
        norm_F = torch.max(torch.norm(F.squeeze(-1), dim=1)).item()
        if norm_F < tol:
            break
            
        bp_eff = -mu * sin_N - c_alpha_batch * (sigma ** 2) * torch.cos(2.0 * N)
        sp_val = sigma * cos_N
        
        J2 = torch.cat([-bp_eff.unsqueeze(-1) * PhiT_batch, PhiT_batch, Zero_m1], dim=2)
        J3 = torch.cat([-sp_val.unsqueeze(-1) * PhiT_batch, Zero_m1, PhiT_batch], dim=2)
        J = torch.cat([J1, J2, J3, bc_row], dim=1)
        
        JtJ = torch.bmm(J.transpose(1, 2), J)
        JtF = torch.bmm(J.transpose(1, 2), -F)
        
        damping = 1e-5 / (1.0 + it)
        diag_JtJ = torch.diagonal(JtJ, dim1=1, dim2=2)
        D = torch.diag_embed(torch.clamp(diag_JtJ, min=1e-6))
        
        dz = torch.linalg.solve(JtJ + damping * D + reg_I, JtF).squeeze(-1)
        step_factor = min(1.0, 0.7 + 0.05 * it)
        z = z + step_factor * dz
        
    c_final = z[:, :m1].numpy()
    Phi_eval = basis_eval(alpha, mhat, t_eval)
    return (c_final @ Phi_eval).squeeze(0) if len(t_eval) == 1 else (c_final @ Phi_eval)

def run_milstein_batch(y0: float, mu: float, sigma: float, dB: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
    n_paths, n_steps = dB.shape
    dt = 1.0 / n_steps
    y = np.full(n_paths, y0, dtype=np.float64)
    eval_indices = np.round(t_eval * n_steps).astype(int)
    eval_pos = 0
    sol = np.zeros((n_paths, len(t_eval)), dtype=np.float64)
    
    if eval_indices[0] == 0:
        sol[:, 0] = y
        eval_pos = 1
        
    for k in range(n_steps):
        dW = dB[:, k]
        sin_y = np.sin(y)
        cos_y = np.cos(y)
        y = y + mu * cos_y * dt + sigma * sin_y * dW + 0.5 * (sigma ** 2) * sin_y * cos_y * (dW ** 2 - dt)
        if eval_pos < len(eval_indices) and k + 1 == eval_indices[eval_pos]:
            sol[:, eval_pos] = y
            eval_pos += 1
            
    return sol

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
    
    out_dir = config.EXPORTS_DIR / "exp5_trig_mu03_sigma015"
    out_dir.mkdir(parents=True, exist_ok=True)
    tex_fig_dir = ROOT / "tex" / "numerics" / "figures"
    tex_fig_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 85)
    print("RUNNING EXPERIMENTAL SUITE: NONLINEAR TRIGONOMETRIC SDE")
    print(f"Parameters: mu={mu}, sigma={sigma}, y0={y0}, R={n_paths:,} paths, N={n_steps:,} steps")
    print("=" * 85)
    
    # 1. Generate Brownian Increments & Milstein Benchmark (alpha = 1.0)
    print(f"Generating Brownian paths ({n_paths:,} paths, {n_steps:,} steps)...")
    rng = np.random.default_rng(config.SEED)
    dB = brownian_paths(n_steps, n_paths, rng=rng, seed=config.SEED)
    
    print("Evaluating high-resolution Milstein benchmark (alpha = 1.0)...")
    t0_bench = time.time()
    exact_eval_a10 = run_milstein_batch(y0, mu, sigma, dB, t_eval)
    print(f">> Milstein benchmark evaluated in {time.time() - t0_bench:.2f}s")
    
    # 2. Classical SDE MSE Convergence Sweep (alpha = 1.0)
    print("\n--- Running Classical SDE MSE Convergence Sweep (alpha = 1.0) ---")
    error_records = []
    
    for m in mhat_values:
        t0 = time.time()
        sol_m = solve_trig_batched_torch(1.0, m, dB, y0, mu, sigma, max(Nq, m + 1), t_eval)
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
    df_errors.to_csv(out_dir / "stochastic_mse_alpha10.csv", index=False)
    
    # 3. Multi-Alpha Mean Error Sweep (mhat = 32)
    print("\n--- Running Multi-Alpha Mean Error Sweep (mhat = 32) ---")
    mean_records = []
    
    for a in alphas_mean:
        print(f"Evaluating mean error for alpha = {a:.2f}...")
        if np.isclose(a, 1.0):
            bench_sol = exact_eval_a10
        else:
            bench_sol = run_fast_fem(MODEL_TRIGONOMETRIC, a, mu, 0.0, sigma, y0, dB, t_eval)
            
        bench_mean = np.mean(bench_sol, axis=0)
        mldnn_sol = solve_trig_batched_torch(a, 32, dB, y0, mu, sigma, Nq, t_eval)
        mldnn_mean = np.mean(mldnn_sol, axis=0)
        
        disc_l2 = float((1.0 / len(t_eval)) * np.sqrt(np.sum((mldnn_mean - bench_mean) ** 2)))
        disc_linf = float(np.max(np.abs(mldnn_mean - bench_mean)))
        mean_records.append({
            "alpha": a,
            "Discrete_L2": disc_l2,
            "Discrete_Linf": disc_linf
        })
        print(f"alpha = {a:.2f} | Discrete L2: {disc_l2:.4e} | Discrete Linf: {disc_linf:.4e}")
        
    df_mean = pd.DataFrame(mean_records)
    df_mean.to_csv(out_dir / "mean_errors_m32.csv", index=False)
    
    # 4. Terminal QQ Plots (alpha = 0.85 & alpha = 1.0, mhat = 32)
    print("\n--- Generating Terminal QQ Plots at t = 1.0 (mhat = 32) ---")
    exact_t1_a10 = exact_eval_a10[:, -1]
    exact_t1_a085 = run_fast_fem(MODEL_TRIGONOMETRIC, 0.85, mu, 0.0, sigma, y0, dB, np.array([1.0])).squeeze(-1)
    
    sol_t1_a10 = solve_trig_batched_torch(1.0, 32, dB, y0, mu, sigma, Nq, np.array([1.0])).squeeze(-1)
    sol_t1_a085 = solve_trig_batched_torch(0.85, 32, dB, y0, mu, sigma, Nq, np.array([1.0])).squeeze(-1)
    
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
        bench_lbl = "Milstein" if np.isclose(a, 1.0) else "fEM"
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
        ax.set_title(f'Nonlinear Trig SDE: {qq_data[a]["title"]}', fontsize=11.5, fontweight='bold')
        ax.set_xlim(q_min - pad, q_max + pad)
        ax.set_ylim(q_min - pad, q_max + pad)
        ax.set_aspect('equal', 'box')
        ax.legend(frameon=True, loc='lower right', fontsize=9.5)
        
    fig_qq.tight_layout()
    qq_comb_path = tex_fig_dir / "qq_trig_alpha085_alpha10_t1_combined.png"
    fig_qq.savefig(qq_comb_path, dpi=300)
    fig_qq.savefig(out_dir / "qq_trig_alpha085_alpha10_t1_combined.png", dpi=300)
    plt.close(fig_qq)
    print(f">> QQ combined plot saved to {qq_comb_path}")
    
    # 5. Sample Path Alpha-Convergence Homotopy Limit Plot
    print("\n--- Generating Sample Path Alpha Convergence Homotopy Limit Plot ---")
    t_eval_fine = np.linspace(0.0, 1.0, 301)
    alphas_homotopy = [0.60, 0.70, 0.80, 0.90, 0.95, 1.00]
    dB_single = brownian_paths(n_steps, 1, seed=config.SEED)
    
    bench_paths = {}
    mldnn_paths = {}
    for a in alphas_homotopy:
        if np.isclose(a, 1.0):
            bench_paths[a] = run_milstein_batch(y0, mu, sigma, dB_single, t_eval_fine)[0]
        else:
            bench_paths[a] = run_fast_fem(MODEL_TRIGONOMETRIC, a, mu, 0.0, sigma, y0, dB_single, t_eval_fine)[0]
        mldnn_paths[a] = solve_trig_batched_torch(a, 32, dB_single, y0, mu, sigma, Nq, t_eval_fine)[0]
        
    colors = {0.60: '#8c564b', 0.70: '#9467bd', 0.80: '#1f77b4', 0.90: '#ff7f0e', 0.95: '#2ca02c', 1.00: '#d62728'}
    fig_sp, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    
    for a in alphas_homotopy:
        lw = 2.4 if np.isclose(a, 1.0) else 1.7
        lbl = u'\u03b1 = 1.00 (Milstein Benchmark)' if np.isclose(a, 1.0) else f'\u03b1 = {a:.2f}'
        ax1.plot(t_eval_fine, bench_paths[a], color=colors[a], linewidth=lw, label=lbl, zorder=5 if np.isclose(a, 1.0) else 3)
        
        lbl_m = u'\u03b1 = 1.00 (MLDNN m = 32)' if np.isclose(a, 1.0) else f'\u03b1 = {a:.2f}'
        ax2.plot(t_eval_fine, mldnn_paths[a], color=colors[a], linewidth=lw, label=lbl_m, zorder=5 if np.isclose(a, 1.0) else 3)
        
    ax1.set_title('(a) High-Resolution Benchmark (N = 65,536)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time t', fontsize=11)
    ax1.set_ylabel(u'State y(t, \u03c9)', fontsize=11)
    ax1.legend(frameon=True, loc='best', fontsize=9.5)
    
    ax2.set_title(u'(b) MLDNN Solution (m = 32)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time t', fontsize=11)
    ax2.legend(frameon=True, loc='best', fontsize=9.5)
    
    fig_sp.tight_layout()
    sp_path = tex_fig_dir / "trig_sample_path_alpha_convergence.png"
    fig_sp.savefig(sp_path, dpi=300)
    fig_sp.savefig(out_dir / "trig_sample_path_alpha_convergence.png", dpi=300)
    plt.close(fig_sp)
    print(f">> Sample path alpha convergence plot saved to {sp_path}")

if __name__ == '__main__':
    main()
