import time
import numpy as np, torch
from solver.core_mldnn import em_caputo, build_S, brownian_increments, coarsen
from solver.solvers import MLDNNSolver

alpha = 0.75
y0, N_paths = 1.0, 500
t_eval = np.linspace(0, 1, 100)

n_gt = 4096
dB_gt = brownian_increments(n_gt * N_paths).reshape(N_paths, n_gt)
gt_sols = em_caputo(alpha, lambda t,y: -y**3, lambda t,y: 1.0, y0, dB_gt)
y_true = np.interp(t_eval, np.linspace(0, 1, n_gt+1), np.mean(gt_sols, axis=0))

m = 16
t0 = time.time()
sols = []
for i in range(N_paths):
    dB_mldnn_i = coarsen(dB_gt[i], 64)
    solver = MLDNNSolver(alpha, m, lambda t,y: -(y**3), lambda t,y: torch.ones_like(y), y0, build_S(alpha, m, dB_mldnn_i), Nq=64)
    solver.train(epochs=20)
    sols.append(solver.evaluate(t_eval))
mldnn_mean = np.mean(sols, axis=0)
t1 = time.time()
print(f"MLDNN m={m}: time={t1-t0:.3f}s, error={np.max(np.abs(y_true - mldnn_mean)):.3e}")

n = 2048
t0 = time.time()
dB_fem = np.array([coarsen(dB_gt[i], n) for i in range(N_paths)])
fem_sols = em_caputo(alpha, lambda t,y: -y**3, lambda t,y: 1.0, y0, dB_fem)
fem_mean = np.interp(t_eval, np.linspace(0, 1, n+1), np.mean(fem_sols, axis=0))
t1 = time.time()
print(f"fEM N={n}: time={t1-t0:.3f}s, error={np.max(np.abs(y_true - fem_mean)):.3e}")
