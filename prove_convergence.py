import numpy as np
from solver.core_mldnn import em_caputo, solve_affine, evaluate_solution, brownian_increments, build_S, coarsen

def prove_convergence():
    alpha = 0.75
    y0, N_paths = 1.0, 500
    t_eval = np.linspace(0, 1, 100)
    
    n_gt = 4096
    dB_gt = brownian_increments(n_gt * N_paths).reshape(N_paths, n_gt)
    
    # 1. Generate the MLELM m=24 Proxy
    gt_sols = []
    for i in range(N_paths):
        S = build_S(alpha, 24, dB_gt[i])
        c, _, _, _ = solve_affine(alpha, 24, S, y0, b0=0.0, b1=1.0, s0=0.0, s1=1.0, Nq=64)
        gt_sols.append(evaluate_solution(alpha, 24, c, t_eval))
    y_true_weak = np.mean(gt_sols, axis=0)
    
    # 2. Check fEM convergence towards the MLELM Proxy
    n_list = [256, 512, 1024, 2048]
    errors = []
    for n in n_list:
        dB_fem = np.array([coarsen(dB_gt[i], n) for i in range(N_paths)])
        fem_sols = em_caputo(alpha, lambda t,y: y, lambda t,y: y, y0, dB_fem)
        fem_mean = np.interp(t_eval, np.linspace(0, 1, n+1), np.mean(fem_sols, axis=0))
        err = np.max(np.abs(y_true_weak - fem_mean))
        errors.append(err)
        
    print("Error of fEM(N) vs MLELM(24):")
    for n, err in zip(n_list, errors):
        print(f"N={n:4d}: {err:.5e}")
        
    print("\nEmpirical Convergence Rate (log2(err_prev / err_curr)):")
    for i in range(1, len(errors)):
        rate = np.log2(errors[i-1] / errors[i])
        print(f"N={n_list[i-1]} -> N={n_list[i]}: Rate = {rate:.3f} (Theoretical: {alpha})")

if __name__ == '__main__':
    np.random.seed(42)
    prove_convergence()
