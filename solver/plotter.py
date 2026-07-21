import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

def plot_experiment_1(t, exact, approx, errors, save_path):
    fig, ax1 = plt.subplots(figsize=(6, 4))
    
    ax1.plot(t, exact, 'r:', label='Exact')
    ax1.plot(t, approx, 'k-', label='MLDNN')
    ax1.legend()
    ax1.set_xlabel('$t$')
    ax1.set_ylabel('$y(t)$')
    ax1.set_title('Solution Comparison')
    
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

def plot_experiment_2(t, mean_mldnn, mean_fem, ci_lower, ci_upper, save_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # ax.plot(t, mean_fem, 'k-', label='fEM Mean')
    ax.plot(t, mean_mldnn, 'k-', label='MLDNN Mean')
    ax.fill_between(t, ci_lower, ci_upper, color='k', alpha=0.2, label='95% CI (MLDNN)')
    
    ax.legend()
    ax.set_xlabel('$t$')
    ax.set_ylabel('Expectation $\\mathbb{E}[X(t)]$')
    ax.set_title('Limit-Relaxation with Additive Noise')
    
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

def plot_experiment_3(t, mean_mldnn, mean_fem, ci_lower, ci_upper, save_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # ax.plot(t, mean_fem, 'k-', label='fEM Mean')
    ax.plot(t, mean_mldnn, 'k-', label='MLDNN Mean')
    ax.fill_between(t, ci_lower, ci_upper, color='k', alpha=0.2, label='95% CI (MLDNN)')
    
    ax.legend()
    ax.set_xlabel('$t$')
    ax.set_ylabel('Expectation $\\mathbb{E}[X(t)]$')
    ax.set_title('Fractional Geometric Brownian Motion (DNN)')
    
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

def save_pareto_table(times_mldnn, err_mldnn, times_fem, err_fem, save_path, labels_mldnn=None, labels_fem=None):
    lines = [
        "| fEM Config | fEM Error | fEM Time (s) | Best MLDNN Config | MLDNN Error | MLDNN Time (s) | Speedup | Memory fEM | Memory MLDNN |",
        "|---|---|---|---|---|---|---|---|---|"
    ]
    
    if not labels_mldnn:
        labels_mldnn = [f"m={i+1}" for i in range(len(times_mldnn))]
    if not labels_fem:
        labels_fem = [f"N={i+1}" for i in range(len(times_fem))]
        
    import re
    
    for tf, ef, lf in zip(times_fem, err_fem, labels_fem):
        # find valid MLDNN configs that beat fEM error
        valid_idxs = [i for i, em in enumerate(err_mldnn) if em <= ef]
        
        if valid_idxs:
            # find the one with the smallest time among valid
            best_idx = min(valid_idxs, key=lambda i: times_mldnn[i])
        else:
            # if none beat it, pick the one with the lowest error
            best_idx = np.argmin(err_mldnn)
            
        tm = times_mldnn[best_idx]
        em = err_mldnn[best_idx]
        lm = labels_mldnn[best_idx]
        
        speedup = tf / tm
        
        # Memory complexity analysis (rough estimate in elements per path)
        try:
            N_match = re.search(r'N=(\d+)', lf)
            N = int(N_match.group(1)) if N_match else 0
        except:
            N = 0
            
        # fEM stores history of size N. MLDNN uses Nq=64.
        mem_fem = f"O(N={N})" if N else "Unknown"
        mem_mldnn = f"O(N_q=64)"
        
        lines.append(f"| {lf} | {ef:.3e} | {tf:.4f} | {lm} | {em:.3e} | {tm:.4f} | **{speedup:.1f}x** | {mem_fem} | {mem_mldnn} |")
        
    with open(save_path, "w") as f:
        f.write("\n".join(lines) + "\n")

def plot_experiment_6(t, true_mean, mean_mldnn, mean_fem, ci_lower, ci_upper, mldnn_label, fem_label, save_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    
    ax.plot(t, true_mean, 'r:', linewidth=2.5, label='Exact Analytical Mean')
    # ax.plot(t, mean_fem, 'k-', alpha=0.7, label=fem_label)
    ax.plot(t, mean_mldnn, 'k-', label=mldnn_label)
    ax.fill_between(t, ci_lower, ci_upper, color='k', alpha=0.2, label=f'95% CI ({mldnn_label})')
    
    ax.legend()
    ax.set_xlabel('$t$')
    ax.set_ylabel('Expectation $\\mathbb{E}[X(t)]$')
    ax.set_title('MLDNN vs fEM: Fast Expectation Tracking')
    
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

def plot_expectation(t, true_mean, mean_mldnn, mean_fem, mldnn_label, fem_label, true_label, title, save_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t, true_mean, 'r:', linewidth=2.5, label=true_label)
    # ax.plot(t, mean_fem, 'k-', alpha=0.7, label=fem_label)
    ax.plot(t, mean_mldnn, 'k-', label=mldnn_label)
    ax.legend(fontsize=9)
    ax.set_xlabel('$t$')
    ax.set_ylabel('Expectation')
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

def plot_variance(t, var_em, var_mldnn, title, save_path):
    """Plot variance over time for EM and MLDNN solutions.
    var_em and var_mldnn are arrays of same length as t.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t, var_em, 'b-', label='EM Variance')
    ax.plot(t, var_mldnn, 'g-', label='MLDNN Variance')
    ax.set_xlabel('$t$')
    ax.set_ylabel('Variance')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

def plot_qq(samples_true, samples_approx, title, save_path):
    import scipy.stats as stats
    fig, ax = plt.subplots(figsize=(5, 5))
    q_true = np.sort(samples_true)
    q_approx = np.sort(samples_approx)
    ax.plot(q_true, q_approx, 'b.', alpha=0.5)
    
    min_val = min(q_true[0], q_approx[0])
    max_val = max(q_true[-1], q_approx[-1])
    ax.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    ax.set_xlabel('Quantiles of True Solution')
    ax.set_ylabel('Quantiles of MLDNN Solution')
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
