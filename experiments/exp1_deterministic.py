import time
import numpy as np
import scipy.special as sc
from solver.core_mldnn import em_caputo, solve_affine, evaluate_solution, ml_vec, build_S
from solver.plotter import plot_expectation, save_pareto_table

def run():
    print("Running Exp 1: Linear Deterministic Pareto")
    alpha = 0.75
    y0 = 1.0
    t_eval = np.linspace(0, 1, 100)
    y_true = ml_vec(alpha, 1.0, -(t_eval ** alpha)) + t_eval**3
    
    m_list = [4, 8, 12, 16, 20, 24]
    times_mldnn, err_mldnn, labels_mldnn = [], [], []
    mldnn_mean_best = None
    
    table_lines = [
        "| $\\hat{m}$ | $L_\\infty$ Error | Time (s) |",
        "|---|---|---|"
    ]
    
    for m in m_list:
        t0 = time.time()
        dB = np.zeros(64)
        S = build_S(alpha, m, dB)
        
        def b0_fun(t):
            return t**3 + sc.gamma(4) / sc.gamma(4 - alpha) * t**(3 - alpha)
            
        c, _, _, _ = solve_affine(alpha, m, S, y0, b0=b0_fun, b1=-1.0, s0=0.0, s1=0.0, Nq=64)
        m_mean = evaluate_solution(alpha, m, c, t_eval)
        t1 = time.time()
        
        dt = t1 - t0
        err = np.max(np.abs(y_true - m_mean))
        
        times_mldnn.append(dt)
        err_mldnn.append(err)
        labels_mldnn.append(f"m={m}")
        
        table_lines.append(f"| {m} | {err:.3e} | {dt:.4f} |")
        
        if m == 24:
            mldnn_mean_best = m_mean
            
    with open("exports/exp1_table.md", "w") as f:
        f.write("\n".join(table_lines) + "\n")
            
    n_list = [256, 512, 1024, 2048, 4096, 8192]
    times_fem, err_fem, labels_fem = [], [], []
    fem_mean_best = None
    
    def bfun_np(t, y): 
        return -y + t**3 + sc.gamma(4) / sc.gamma(4 - alpha) * t**(3 - alpha)
        
    def sfun_np(t, y): 
        return 0.0
    
    for n in n_list:
        t0 = time.time()
        fem_sol = em_caputo(alpha, bfun_np, sfun_np, y0, np.zeros(n))
        fem_mean = np.interp(t_eval, np.linspace(0, 1, n+1), fem_sol)
        t1 = time.time()
        
        times_fem.append(t1 - t0)
        err_fem.append(np.max(np.abs(y_true - fem_mean)))
        labels_fem.append(f"N={n}")
        if n == 8192:
            fem_mean_best = fem_mean
            
    plot_expectation(t_eval, y_true, mldnn_mean_best, fem_mean_best, f"MLELM ($\\hat{{m}}$={m_list[-1]})", f"fEM (N={n_list[-1]})", "Exact Analytical", "Example 4.1", "exports/exp1_deterministic.pdf")
    save_pareto_table(times_mldnn, err_mldnn, times_fem, err_fem, "exports/exp1_pareto.md", labels_mldnn, labels_fem)
    print(f"Saved exp1!\n")

if __name__ == '__main__':
    run()
