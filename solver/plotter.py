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

def plot_pareto(times_mldnn, err_mldnn, times_fem, err_fem, save_path, labels_mldnn=None, labels_fem=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    
    ax.loglog(times_mldnn, err_mldnn, 'ko-', label='MLDNN')
    ax.loglog(times_fem, err_fem, 'bs-', label='fEM')
    
    if labels_mldnn:
        for i, txt in enumerate(labels_mldnn):
            ax.annotate(txt, (times_mldnn[i], err_mldnn[i]), xytext=(5, 5), textcoords='offset points', color='red', fontsize=8)
            
    if labels_fem:
        for i, txt in enumerate(labels_fem):
            ax.annotate(txt, (times_fem[i], err_fem[i]), xytext=(5, -12), textcoords='offset points', color='black', fontsize=8)
    
    ax.legend()
    ax.set_xlabel('Computational Time (s)')
    ax.set_ylabel('Error Threshold')
    ax.set_title('Pareto Efficiency')
    
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

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
