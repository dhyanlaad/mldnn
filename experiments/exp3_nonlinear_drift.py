import numpy as np, torch
from solver.core_mldnn import em_caputo, build_S, brownian_increments
from solver.solvers import MLDNNSolver
from solver.plotter import plot_expectation

def run():
    print("Running Exp 3: Nonlinear Drift")
    alpha = 0.75
    y0, N_paths = 1.0, 500
    t_eval = np.linspace(0, 1, 100)
    
    n_gt = 1024
    dB_gt = brownian_increments(n_gt * N_paths).reshape(N_paths, n_gt)
    gt_sols = [em_caputo(alpha, lambda t,y: -y**3, lambda t,y: 1.0, y0, dB_gt[i]) for i in range(N_paths)]
    y_true = np.interp(t_eval, np.linspace(0, 1, n_gt+1), np.mean(gt_sols, axis=0))
    
    m = 16
    dB_mldnn = brownian_increments(64 * N_paths).reshape(N_paths, 64)
    sols = []
    for i in range(N_paths):
        solver = MLDNNSolver(alpha, m, lambda t,y: -(y**3), lambda t,y: torch.ones_like(y), y0, build_S(alpha, m, dB_mldnn[i]), Nq=64)
        solver.train(epochs=20)
        sols.append(solver.evaluate(t_eval))
    mldnn_mean = np.mean(sols, axis=0)
        
    n = 32
    dB_fem = brownian_increments(n * N_paths).reshape(N_paths, n)
    fem_sols = [em_caputo(alpha, lambda t,y: -y**3, lambda t,y: 1.0, y0, dB_fem[i]) for i in range(N_paths)]
    fem_mean = np.interp(t_eval, np.linspace(0, 1, n+1), np.mean(fem_sols, axis=0))
    
    plot_expectation(t_eval, y_true, mldnn_mean, fem_mean, f"MLDNN (DNN, m={m})", f"fEM (N={n})", "Ground Truth", "Exp 3: Nonlinear Drift", "exp3_nonlinear_drift.pdf")
    print(f"Saved exp3!\n")

if __name__ == '__main__': np.random.seed(42); torch.manual_seed(42); run()
