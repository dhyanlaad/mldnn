"""
exp1_deterministic.py
=====================
Experiment 1: Deterministic Fractional Relaxation (Zero Diffusion)
Equation: D_t^alpha X(t) = -theta * X(t), X(0) = 1.0, theta = 1.0, sigma = 0.
Exact Analytic Solution: X(t) = X0 * E_alpha(-theta * t^alpha)
Sweeps:
- alpha in {0.55, 0.60, 0.70, 0.80, 0.90, 1.00}
- mhat in {2, 4, 8, 16, 24, 32, 40}
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from solver.core_mldnn import ml_vec, solve_affine, evaluate_solution
from experiments.common import save_experiment_cache

def run_experiment_1():
    print("=" * 80)
    print("Running Experiment 1: Deterministic Benchmark (Zero Diffusion)")
    print("Equation: D_t^alpha X(t) = -theta * X(t), theta = 1.0, X(0) = 1.0")
    print("=" * 80)
    
    out_dir = config.EXPORTS_DIR / "exp1_deterministic"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    alphas = [0.55, 0.60, 0.70, 0.80, 0.90, 1.00]
    mhat_values = [2, 4, 8, 16, 24, 32, 40]
    Nq = 64
    theta = 1.0
    x0 = 1.0
    
    t_eval = np.linspace(0.0, 1.0, 1001)
    
    l2_errors = {m: {} for m in mhat_values}
    max_abs_errors = {m: {} for m in mhat_values}
    mean_abs_errors = {m: {} for m in mhat_values}
    sup_mse_errors = {m: {} for m in mhat_values}
    term_errors = {m: {} for m in mhat_values}
    
    exact_dict = {}
    mldnn_dict = {a: {} for a in alphas}
    
    for alpha in alphas:
        print(f"\n>>> Processing alpha = {alpha:.2f} <<<")
        exact = x0 * ml_vec(alpha, 1.0, -theta * (t_eval ** alpha))
        exact_dict[alpha] = exact[None, :]
        
        for mhat in mhat_values:
            Nq_eff = max(Nq, mhat + 1)
            c, tb, ts, _ = solve_affine(
                alpha=alpha,
                mhat=mhat,
                S=None,
                y0=x0,
                b0=0.0,
                b1=-theta,
                s0=0.0,
                s1=0.0,
                Nq=Nq_eff
            )
            y_pred = evaluate_solution(alpha, mhat, c, t_eval)
            mldnn_dict[alpha][mhat] = y_pred[None, :]
            
            diff = exact - y_pred
            abs_diff = np.abs(diff)
            
            l2_err = float(np.sqrt(np.trapz(diff ** 2, t_eval)))
            max_abs = float(np.max(abs_diff))
            mean_abs = float(np.mean(abs_diff))
            sup_mse = float(np.max(diff ** 2))
            term_err = float(abs_diff[-1])
            
            l2_errors[mhat][str(alpha)] = l2_err
            max_abs_errors[mhat][str(alpha)] = max_abs
            mean_abs_errors[mhat][str(alpha)] = mean_abs
            sup_mse_errors[mhat][str(alpha)] = sup_mse
            term_errors[mhat][str(alpha)] = term_err
            
            print(f"alpha = {alpha:.2f} | mhat = {mhat:2d} | L2 Error: {l2_err:.4e} | Max Abs: {max_abs:.4e} | Mean Abs: {mean_abs:.4e}")

    # Export CSVs
    df_l2 = pd.DataFrame(l2_errors).T
    df_max_abs = pd.DataFrame(max_abs_errors).T
    df_mean_abs = pd.DataFrame(mean_abs_errors).T
    df_sup_mse = pd.DataFrame(sup_mse_errors).T
    df_term = pd.DataFrame(term_errors).T
    
    df_l2.index.name = "mhat"
    df_max_abs.index.name = "mhat"
    df_mean_abs.index.name = "mhat"
    df_sup_mse.index.name = "mhat"
    df_term.index.name = "mhat"
    
    df_l2.to_csv(out_dir / "errors_l2.csv")
    df_max_abs.to_csv(out_dir / "errors_max_abs.csv")
    df_mean_abs.to_csv(out_dir / "errors_mean_abs.csv")
    df_sup_mse.to_csv(out_dir / "errors_sup_mse.csv")
    df_term.to_csv(out_dir / "errors_terminal.csv")
    
    # Save complete cache
    metrics_dict = {
        "L2": l2_errors,
        "Max_Abs": max_abs_errors,
        "Mean_Abs": mean_abs_errors,
        "Sup_MSE": sup_mse_errors,
        "Terminal": term_errors
    }
    save_experiment_cache(
        out_dir=out_dir,
        t_eval=t_eval,
        alphas=alphas,
        mhat_values=mhat_values,
        exact_dict=exact_dict,
        mldnn_dict=mldnn_dict,
        metrics_dict=metrics_dict,
        params={"theta": theta, "x0": x0, "Nq": Nq, "model": "OU zero diffusion"}
    )
    
    # Markdown tables
    def make_md_table(df_table: pd.DataFrame, metric_name: str) -> str:
        headers = ["m_hat"] + [f"alpha = {a:.2f}" for a in alphas]
        lines = [
            f"# Experiment 1: Deterministic Fractional Relaxation ({metric_name})",
            "",
            f"Equation: D_t^alpha X(t) = -{theta} X(t), X(0) = {x0}",
            f"Benchmark: Exact Analytic Mittag-Leffler Solution X(t) = E_alpha(-t^alpha)",
            "",
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join([":---:"] * len(headers)) + " |"
        ]
        for m in mhat_values:
            row = [f"{m:2d}"] + [f"{df_table.loc[m, str(a)]:.4e}" for a in alphas]
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    md_l2 = make_md_table(df_l2, "L2 Error")
    md_max_abs = make_md_table(df_max_abs, "Maximum Absolute Error (L_inf)")
    md_mean_abs = make_md_table(df_mean_abs, "Mean Absolute Error")

    with open(out_dir / "table_l2_errors.md", "w") as f:
        f.write(md_l2)
        f.write("\n")
    with open(out_dir / "table_max_abs_errors.md", "w") as f:
        f.write(md_max_abs)
        f.write("\n")
    with open(out_dir / "table_mean_abs_errors.md", "w") as f:
        f.write(md_mean_abs)
        f.write("\n")
        
    # Plot Spectral Convergence (Log-Linear)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    cmap = plt.get_cmap("viridis")
    for idx, a in enumerate(alphas):
        color = cmap(idx / len(alphas))
        ax.semilogy(mhat_values, df_l2[str(a)], marker='o', label=rf"$\alpha = {a:.2f}$", color=color, linewidth=1.8)
    ax.set_xlabel(r"Spectral Basis Degree $\hat{m}$", fontsize=12)
    ax.set_ylabel(r"$L_2$ Approximation Error", fontsize=12)
    ax.set_title(r"Deterministic Fractional Relaxation: Exponential Convergence in $\hat{m}$", fontsize=13, fontweight='bold')
    ax.set_xticks(mhat_values)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, which="both", alpha=0.3, ls="--")
    fig.tight_layout()
    fig.savefig(out_dir / "convergence_all_alphas.png", dpi=300)
    plt.close(fig)
    
    # Plot Trajectories
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for a in [0.55, 0.70, 0.90, 1.00]:
        ax.plot(t_eval, exact_dict[a][0], '-', label=rf"Exact $\alpha={a:.2f}$", linewidth=2.0)
        ax.plot(t_eval, mldnn_dict[a][40][0], '--', label=rf"MLDNN $\hat{{m}}=40, \alpha={a:.2f}$", linewidth=1.5)
    ax.set_xlabel("Time $t$", fontsize=12)
    ax.set_ylabel("$X(t)$", fontsize=12)
    ax.set_title("Deterministic Relaxation Trajectories vs Exact Mittag-Leffler", fontsize=13, fontweight='bold')
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3, ls="--")
    fig.tight_layout()
    fig.savefig(out_dir / "trajectories.png", dpi=300)
    plt.close(fig)
    
    print("\n" + "=" * 80)
    print("EXPERIMENT 1 COMPLETE! SUMMARY TABLES:")
    print("=" * 80)
    print("\n--- L2 ERROR TABLE ---")
    print(df_l2.to_markdown())
    print("\n--- MAXIMUM ABSOLUTE ERROR TABLE (L_inf) ---")
    print(df_max_abs.to_markdown())
    print("\n--- MEAN ABSOLUTE ERROR TABLE ---")
    print(df_mean_abs.to_markdown())
    print(f"\nArtifacts and cache saved to: {out_dir}")

if __name__ == "__main__":
    run_experiment_1()
