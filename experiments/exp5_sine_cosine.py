import numpy as np, torch
from solver.core_mldnn import em_caputo, build_S, brownian_increments
from solver.solvers import MLDNNSolver
from solver.plotter import plot_expectation

def run():
    print("Running Exp 5: Sine-Cosine")
    alpha = 0.75
    y0, N_paths = 1.0, 500
    t_eval = np.linspace(0, 1, 100)
    
    n_gt = 1024
    dB_gt = brownian_increments(n_gt * N_paths).reshape(N_paths, n_gt)
    gt_sols = [em_caputo(alpha, lambda t,y: np.sin(y), lambda t,y: np.cos(y), y0, dB_gt[i]) for i in range(N_paths)]
    y_true = np.interp(t_eval, np.linspace(0, 1, n_gt+1), np.mean(gt_sols, axis=0))
    
    m = 8
    dB_mldnn = brownian_increments(64 * N_paths).reshape(N_paths, 64)
    sols = []
    for i in range(N_paths):
        solver = MLDNNSolver(alpha, m, lambda t,y: torch.sin(y), lambda t,y: torch.cos(y), y0, build_S(alpha, m, dB_mldnn[i]), Nq=64)
        solver.train(epochs=20)
        sols.append(solver.evaluate(t_eval))
    mldnn_mean = np.mean(sols, axis=0)
        
    n = 4096
    dB_fem = brownian_increments(n * N_paths).reshape(N_paths, n)
    fem_sols = [em_caputo(alpha, lambda t,y: np.sin(y), lambda t,y: np.cos(y), y0, dB_fem[i]) for i in range(N_paths)]
    fem_mean = np.interp(t_eval, np.linspace(0, 1, n+1), np.mean(fem_sols, axis=0))
    
    plot_expectation(t_eval, y_true, mldnn_mean, fem_mean, f"MLDNN ($\\hat{{m}}$ = {m})", f"fEM (N={n})", "Ground truth", "Example 4.5", "exp5_sine_cosine.pdf")
    print(f"Saved exp5!\n")

if __name__ == '__main__': np.random.seed(42); torch.manual_seed(42); run()
