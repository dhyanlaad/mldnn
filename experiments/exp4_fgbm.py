import numpy as np, torch
from solver.core_mldnn import em_caputo, build_S, brownian_increments, ml_vec
from solver.solvers import MLDNNSolver
from solver.plotter import plot_expectation

def run():
    print("Running Exp 4: fGBM")
    alpha = 0.75
    y0, N_paths = 1.0, 500
    t_eval = np.linspace(0, 1, 100)
    y_true = ml_vec(alpha, 1.0, t_eval ** alpha)
    
    m = 12
    dB_mldnn = brownian_increments(64 * N_paths).reshape(N_paths, 64)
    sols = []
    for i in range(N_paths):
        solver = MLDNNSolver(alpha, m, lambda t,y: y, lambda t,y: y, y0, build_S(alpha, m, dB_mldnn[i]), Nq=64)
        solver.train(epochs=20)
        sols.append(solver.evaluate(t_eval))
    mldnn_mean = np.mean(sols, axis=0)
        
    n = 64
    dB_fem = brownian_increments(n * N_paths).reshape(N_paths, n)
    fem_sols = [em_caputo(alpha, lambda t,y: y, lambda t,y: y, y0, dB_fem[i]) for i in range(N_paths)]
    fem_mean = np.interp(t_eval, np.linspace(0, 1, n+1), np.mean(fem_sols, axis=0))
    
    plot_expectation(t_eval, y_true, mldnn_mean, fem_mean, f"MLDNN (DNN, m={m})", f"fEM (N={n})", "Exact Analytic", "Exp 4: Fractional GBM", "exp4_fgbm.pdf")
    print(f"Saved exp4!\n")

if __name__ == '__main__': np.random.seed(42); torch.manual_seed(42); run()
