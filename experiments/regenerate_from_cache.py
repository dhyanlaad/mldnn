"""
regenerate_from_cache.py
=========================
Regenerates every plot produced by exp1-exp5 directly from the cached
simulation data in `cache/`, with NO re-simulation (no C fEM calls, no
Gauss-Newton solves). Writes all figures into `final_img/<exp_name>/`.

Where a plot has no raw numeric cache available (exp5_trig has none - only
CSV error summaries were persisted, the raw sample paths were never saved),
the already-rendered PNG sitting in cache/ is copied through unchanged
instead of being silently skipped.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
OUT = ROOT / "final_img"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.8,
    "savefig.dpi": 300
})


def qq_combined_plot(bench_a, sol_a, bench_b, sol_b, label_a, label_b,
                      bench_lbl_a, bench_lbl_b, process_name, save_path):
    """Reproduces the side-by-side QQ plot code shared by exp2/exp3/exp4, with
    titles removed and axis labels standardized across every QQ plot. Only the
    alpha value is annotated (no mu/sigma statistics), and the legend uses the
    manuscript's ML-PIFLNN / n notation."""
    probs = np.linspace(0.005, 0.995, 200)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))
    axes = {label_a: ax1, label_b: ax2}
    qq_data = {
        label_a: {"bench": bench_a, "sol": sol_a, "lbl": bench_lbl_a},
        label_b: {"bench": bench_b, "sol": sol_b, "lbl": bench_lbl_b},
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

        ax.plot(line_vals, line_vals, color='#7f7f7f', linestyle='--', linewidth=1.4,
                label='Reference ($y = x$)', zorder=1)
        ax.scatter(q_bench, q_sol, color='#1f77b4', alpha=0.7, s=18, edgecolors='none',
                   label=r'ML-PIFLNN $n = 32$', zorder=3)

        ax.text(0.05, 0.92, f"$\\alpha = {a:.2f}$", transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='gray'))

        ax.set_xlabel('Benchmark Quantiles', fontsize=11)
        ax.set_ylabel('ML-PIFLNN Quantiles', fontsize=11)
        ax.set_xlim(q_min - pad, q_max + pad)
        ax.set_ylim(q_min - pad, q_max + pad)
        ax.set_aspect('equal', 'box')
        ax.legend(frameon=True, loc='lower right', fontsize=9.5)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  -> {save_path}")


# ---------------------------------------------------------------------------
# Experiment 1: Deterministic Fractional Relaxation
# ---------------------------------------------------------------------------
def regen_exp1():
    print("=" * 70)
    print("Experiment 1: Deterministic Fractional Relaxation (from cache)")
    print("=" * 70)
    src = CACHE / "exp1_deterministic"
    dst = OUT / "exp1_deterministic"
    dst.mkdir(parents=True, exist_ok=True)

    data = np.load(src / "data_cache.npz")
    t_eval = data["t_eval"]
    alphas = [float(a) for a in data["alphas"]]
    mhat_values = [int(m) for m in data["mhat_values"]]

    def key(a):
        return f"{a}" if a != int(a) else f"{a}"

    exact_dict = {a: data[f"exact_alpha_{a}"] for a in alphas}
    mldnn_dict = {a: {m: data[f"mldnn_alpha_{a}_m_{m}"] for m in mhat_values} for a in alphas}

    df_l2 = pd.read_csv(src / "errors_l2.csv", index_col="mhat")
    df_l2.columns = [float(c) for c in df_l2.columns]

    # --- Spectral convergence plot ---
    fig, ax = plt.subplots(figsize=(8.5, 5))
    cmap = plt.get_cmap("viridis")
    for idx, a in enumerate(alphas):
        color = cmap(idx / len(alphas))
        ax.semilogy(mhat_values, [df_l2.loc[m, a] for m in mhat_values], marker='o',
                    label=rf"$\alpha = {a:.2f}$", color=color, linewidth=1.8)
    ax.set_xlabel(r"$\hat{m}$", fontsize=12)
    ax.set_ylabel(r"$L_2$ Error", fontsize=12)
    ax.set_xticks(mhat_values)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, which="both", alpha=0.3, ls="--")
    fig.tight_layout()
    fig.savefig(dst / "convergence_all_alphas.png", dpi=300)
    plt.close(fig)
    print(f"  -> {dst / 'convergence_all_alphas.png'}")

    # --- Trajectories plot ---
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for a in [0.55, 0.70, 0.90, 1.00]:
        ax.plot(t_eval, exact_dict[a][0], '-', label=rf"Exact $\alpha={a:.2f}$", linewidth=2.0)
        ax.plot(t_eval, mldnn_dict[a][40][0], '--', label=rf"ML-PIFLNN $n=40, \alpha={a:.2f}$", linewidth=1.5)
    ax.set_xlabel(r"Time $t$", fontsize=12)
    ax.set_ylabel(r"$y(t)$", fontsize=12)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3, ls="--")
    fig.tight_layout()
    fig.savefig(dst / "trajectories.png", dpi=300)
    plt.close(fig)
    print(f"  -> {dst / 'trajectories.png'}")


# ---------------------------------------------------------------------------
# Experiment 2: Stochastic Ornstein-Uhlenbeck
# ---------------------------------------------------------------------------
def regen_exp2():
    print("=" * 70)
    print("Experiment 2: Stochastic Ornstein-Uhlenbeck (from cache)")
    print("=" * 70)
    src = CACHE / "exp2_ou"
    dst = OUT / "exp2_ou"
    dst.mkdir(parents=True, exist_ok=True)

    data = np.load(src / "data_cache.npz")
    t_eval = data["t_eval"]
    mhat_values = [int(m) for m in data["mhat_values"]]
    exact = data["exact_alpha_1.0"]

    # --- trajectories_m1_m40: single representative path across increasing mhat ---
    path_idx = 0
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(t_eval, exact[path_idx], 'k--', linewidth=2.2, label="Dense Milstein Benchmark", zorder=5)
    cmap = plt.get_cmap("plasma")
    show_m = [m for m in mhat_values if m in (1, 2, 4, 8, 16, 24, 32, 40)]
    for idx, m in enumerate(show_m):
        color = cmap(idx / max(1, len(show_m) - 1))
        sol = data[f"mldnn_alpha_1.0_m_{m}"]
        ax.plot(t_eval, sol[path_idx], '-', color=color, linewidth=1.5, alpha=0.85,
                label=rf"ML-PIFLNN $n={m}$")
    ax.set_xlabel(r"Time $t$", fontsize=12)
    ax.set_ylabel(r"$y(t)$", fontsize=12)
    ax.legend(frameon=True, fontsize=8.5, ncol=2)
    fig.tight_layout()
    fig.savefig(dst / "trajectories_m1_m40.png", dpi=300)
    plt.close(fig)
    print(f"  -> {dst / 'trajectories_m1_m40.png'}")

    # --- QQ plots from raw terminal-value cache ---
    # Note: the "alpha07" filename is retained for compatibility with the
    # existing \includegraphics path in main.tex (Figure 1), but per request
    # it is now populated with the alpha=0.85 panel (identical to Figure 2)
    # rather than alpha=0.70.
    qq = np.load(src / "qq_raw_cache.npz")
    qq_combined_plot(
        bench_a=qq["exact_t1_a085"], sol_a=qq["mldnn_t1_a085"],
        bench_b=qq["exact_t1_a10"], sol_b=qq["mldnn_t1_a10"],
        label_a=0.85, label_b=1.00,
        bench_lbl_a="fEM", bench_lbl_b="Exact",
        process_name="Fractional OU",
        save_path=dst / "qq_ou_alpha07_alpha10_t1_combined.png"
    )
    qq_combined_plot(
        bench_a=qq["exact_t1_a085"], sol_a=qq["mldnn_t1_a085"],
        bench_b=qq["exact_t1_a10"], sol_b=qq["mldnn_t1_a10"],
        label_a=0.85, label_b=1.00,
        bench_lbl_a="fEM", bench_lbl_b="Exact",
        process_name="Fractional OU",
        save_path=dst / "qq_ou_alpha085_alpha10_t1_combined.png"
    )


# ---------------------------------------------------------------------------
# Experiment 3: Geometric Brownian Motion
# ---------------------------------------------------------------------------
def regen_exp3():
    print("=" * 70)
    print("Experiment 3: Geometric Brownian Motion (from cache)")
    print("=" * 70)
    src = CACHE / "exp3_gbm"
    dst = OUT / "exp3_gbm"
    dst.mkdir(parents=True, exist_ok=True)

    data = np.load(src / "data_cache.npz")
    t_eval = data["t_eval"]
    exact = data["exact_alpha_1.0"]
    mldnn_m1 = data["mldnn_alpha_1.0_m_1"]

    # --- gbm_sample_paths_m1_comparison: single representative path, mhat=1 vs Milstein ---
    path_idx = 0
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(t_eval, exact[path_idx], 'k--', linewidth=2.0, label="Dense Milstein Benchmark", zorder=5)
    ax.plot(t_eval, mldnn_m1[path_idx], '-', color='#1f77b4', linewidth=1.8,
            label=r"ML-PIFLNN $n=1$")
    ax.set_xlabel(r"Time $t$", fontsize=12)
    ax.set_ylabel(r"$y(t)$", fontsize=12)
    ax.legend(frameon=True, fontsize=10)
    fig.tight_layout()
    fig.savefig(dst / "gbm_sample_paths_m1_comparison.png", dpi=300)
    plt.close(fig)
    print(f"  -> {dst / 'gbm_sample_paths_m1_comparison.png'}")

    # --- Trace-correction demonstration: single realization, corrected vs. uncorrected ---
    tc = np.load(src / "trace_correction_single_demo_cache.npz")
    t_tc = tc["t_eval"]
    bench_tc = tc["benchmark"]
    corrected = tc["corrected"]
    uncorrected = tc["uncorrected"]

    fig, (ax_c, ax_u) = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, sol, sol_label in [(ax_c, corrected, "ML-PIFLNN (trace-corrected)"),
                                (ax_u, uncorrected, "ML-PIFLNN (uncorrected)")]:
        ax.plot(t_tc, bench_tc, 'k--', linewidth=2.0, label="Exact Benchmark", zorder=5)
        ax.plot(t_tc, sol, '-', color='#1f77b4', linewidth=1.8, label=sol_label)
        ax.set_xlabel(r"Time $t$", fontsize=11)
        ax.set_ylabel(r"$y(t)$", fontsize=11)
        ax.legend(frameon=True, loc="best", fontsize=9.5)

    fig.tight_layout()
    fig.savefig(dst / "gbm_trace_correction_comparison.png", dpi=300)
    plt.close(fig)
    print(f"  -> {dst / 'gbm_trace_correction_comparison.png'}")

    # --- gbm_sample_path_alpha_convergence: bench vs mldnn, fractional homotopy ---
    sp = np.load(src / "sample_path_alpha_convergence_cache.npz")
    t_fine = sp["t_eval"]
    alphas_homotopy = [0.75, 0.85, 0.95, 1.00]
    colors = {0.75: '#1f77b4', 0.85: '#ff7f0e', 0.95: '#2ca02c', 1.00: '#d62728'}

    fig, (ax_fem, ax_mldnn) = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for a in alphas_homotopy:
        bench = sp[f"bench_alpha_{a}"]
        mldnn = sp[f"mldnn_alpha_{a}"]
        if a == 1.00:
            ax_fem.plot(t_fine, bench, color=colors[a], linewidth=2.4, label=r'$\alpha = 1.00$ (Milstein Benchmark)', zorder=5)
            ax_mldnn.plot(t_fine, mldnn, color=colors[a], linewidth=2.4, label=r'$\alpha = 1.00$ (ML-PIFLNN $n = 32$)', zorder=5)
        else:
            ax_fem.plot(t_fine, bench, color=colors[a], linewidth=1.8, label=rf'$\alpha = {a:.2f}$')
            ax_mldnn.plot(t_fine, mldnn, color=colors[a], linewidth=1.8, label=rf'$\alpha = {a:.2f}$')

    ax_fem.set_xlabel(r'Time $t$', fontsize=11)
    ax_fem.set_ylabel(r'$y(t)$', fontsize=11)
    ax_fem.legend(frameon=True, loc='best', fontsize=10)

    ax_mldnn.set_xlabel(r'Time $t$', fontsize=11)
    ax_mldnn.set_ylabel(r'$y(t)$', fontsize=11)
    ax_mldnn.legend(frameon=True, loc='best', fontsize=10)

    fig.tight_layout()
    fig.savefig(dst / "gbm_sample_path_alpha_convergence.png", dpi=300)
    plt.close(fig)
    print(f"  -> {dst / 'gbm_sample_path_alpha_convergence.png'}")

    # --- QQ plots ---
    # Note: as with exp2, the "alpha07" filename is kept for compatibility
    # with main.tex (Figure 3), but now holds the alpha=0.85 panel.
    qq = np.load(src / "qq_raw_cache.npz")
    qq_combined_plot(
        bench_a=qq["exact_t1_a085"], sol_a=qq["mldnn_t1_a085"],
        bench_b=qq["exact_t1_a10"], sol_b=qq["mldnn_t1_a10"],
        label_a=0.85, label_b=1.00,
        bench_lbl_a="fEM", bench_lbl_b="Exact",
        process_name="Fractional GBM",
        save_path=dst / "qq_gbm_alpha07_alpha10_t1_combined.png"
    )
    qq_combined_plot(
        bench_a=qq["exact_t1_a085"], sol_a=qq["mldnn_t1_a085"],
        bench_b=qq["exact_t1_a10"], sol_b=qq["mldnn_t1_a10"],
        label_a=0.85, label_b=1.00,
        bench_lbl_a="fEM", bench_lbl_b="Exact",
        process_name="Fractional GBM",
        save_path=dst / "qq_gbm_alpha085_alpha10_t1_combined.png"
    )


# ---------------------------------------------------------------------------
# Experiment 4: Cox-Ingersoll-Ross (square-root diffusion)
# ---------------------------------------------------------------------------
def regen_exp4():
    print("=" * 70)
    print("Experiment 4: Cox-Ingersoll-Ross Process (from cache)")
    print("=" * 70)
    src = CACHE / "exp4_cir"
    dst = OUT / "exp4_cir"
    dst.mkdir(parents=True, exist_ok=True)

    qq = np.load(src / "qq_raw_cache.npz")
    qq_combined_plot(
        bench_a=qq["exact_t1_a085"], sol_a=qq["mldnn_t1_a085"],
        bench_b=qq["exact_t1_a10"], sol_b=qq["mldnn_t1_a10"],
        label_a=0.85, label_b=1.00,
        bench_lbl_a="fEM", bench_lbl_b="fEM",
        process_name="Fractional CIR",
        save_path=dst / "qq_cir_alpha085_alpha10_t1_combined.png"
    )


# ---------------------------------------------------------------------------
# Experiment 5: Nonlinear Trigonometric SDE
# Both the QQ plot and the sample-path homotopy plot now have real raw-data
# caches (qq_raw_cache.npz and sample_path_alpha_convergence_cache.npz),
# computed live once via the Milstein/fEM benchmarks and the Gauss-Newton
# ML-PIFLNN solver, since neither cache previously existed and the old
# rendered PNGs had a broken (garbled) axis-label / stale-branding raster.
# ---------------------------------------------------------------------------
def regen_exp5():
    print("=" * 70)
    print("Experiment 5: Nonlinear Trigonometric SDE")
    print("=" * 70)
    src = CACHE / "exp5_trig"
    dst = OUT / "exp5_trig"
    dst.mkdir(parents=True, exist_ok=True)

    qq = np.load(src / "qq_raw_cache.npz")
    qq_combined_plot(
        bench_a=qq["exact_t1_a085"], sol_a=qq["mldnn_t1_a085"],
        bench_b=qq["exact_t1_a10"], sol_b=qq["mldnn_t1_a10"],
        label_a=0.85, label_b=1.00,
        bench_lbl_a="fEM", bench_lbl_b="Milstein",
        process_name="Nonlinear Trigonometric SDE",
        save_path=dst / "qq_trig_alpha085_alpha10_t1_combined.png"
    )

    # --- Sample-path alpha-convergence homotopy plot ---
    sp = np.load(src / "sample_path_alpha_convergence_cache.npz")
    t_fine = sp["t_eval"]
    alphas_homotopy = [0.60, 0.70, 0.80, 0.90, 0.95, 1.00]
    colors = {0.60: '#8c564b', 0.70: '#9467bd', 0.80: '#1f77b4',
              0.90: '#ff7f0e', 0.95: '#2ca02c', 1.00: '#d62728'}

    fig, (ax_bench, ax_mldnn) = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for a in alphas_homotopy:
        bench = sp[f"bench_alpha_{a}"]
        mldnn = sp[f"mldnn_alpha_{a}"]
        if a == 1.00:
            ax_bench.plot(t_fine, bench, color=colors[a], linewidth=2.4,
                          label=r'$\alpha = 1.00$ (Milstein Benchmark)', zorder=5)
            ax_mldnn.plot(t_fine, mldnn, color=colors[a], linewidth=2.4,
                          label=r'$\alpha = 1.00$ (ML-PIFLNN $n = 32$)', zorder=5)
        else:
            ax_bench.plot(t_fine, bench, color=colors[a], linewidth=1.7, label=rf'$\alpha = {a:.2f}$')
            ax_mldnn.plot(t_fine, mldnn, color=colors[a], linewidth=1.7, label=rf'$\alpha = {a:.2f}$')

    ax_bench.set_xlabel(r'Time $t$', fontsize=11)
    ax_bench.set_ylabel(r'$y(t)$', fontsize=11)
    ax_bench.legend(frameon=True, loc='best', fontsize=9.5)

    ax_mldnn.set_xlabel(r'Time $t$', fontsize=11)
    ax_mldnn.set_ylabel(r'$y(t)$', fontsize=11)
    ax_mldnn.legend(frameon=True, loc='best', fontsize=9.5)

    fig.tight_layout()
    fig.savefig(dst / "trig_sample_path_alpha_convergence.png", dpi=300)
    plt.close(fig)
    print(f"  -> {dst / 'trig_sample_path_alpha_convergence.png'}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    regen_exp1()
    regen_exp2()
    regen_exp3()
    regen_exp4()
    regen_exp5()
    print("\nAll figures regenerated into:", OUT)
