"""
parallel.py
===========
High-performance parallelized solver routines for multi-core Apple Silicon / Mac CPUs.
Integrates the Stochastic Fubini Adjoint Tensor contraction for S_alpha, enabling
instantaneous (50,000+ paths/sec) S_alpha construction and batched linear algebraic solves.
"""

from __future__ import annotations
import os
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor
from scipy.special import gamma as sgamma
from scipy.linalg import lstsq
from .core_mldnn import (
    get_A,
    basis_eval,
    chebyshev_nodes,
    _pow_diff,
    _causal_conv,
    evaluate_solution,
    trace_coefficient,
    prepare_operator_trace_quadrature,
)

def build_fubini_tensor(alpha: float, mhat: int, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct the deterministic Stochastic Fubini Adjoint Contraction Tensor M_tens.
    By Stochastic Fubini, the cross-covariance Xi_ik = int_0^1 K_k(tau) M_i(tau) dB_tau
    is an exact linear map from the Brownian increments dB to the matrix Xi.
    
    Returns:
        M_tens: ( (mhat+1)^2, n_steps ) matrix such that (M_tens @ dB.T).T gives flattened Xi.
        omega_inv: diagonal weights (2*alpha*k + 1)
    """
    tm = np.linspace(0.0, 1.0, n_steps + 1)
    Mmesh = basis_eval(alpha, mhat, tm)
    m1 = mhat + 1
    D = 1.0 / n_steps
    d = np.arange(1, n_steps + 1, dtype=float)
    with np.errstate(divide="ignore"):
        pdiff_a = -np.expm1(alpha * np.log1p(-1.0 / d)) * np.power(d, alpha)
        pdiff_a1 = -np.expm1((alpha + 1.0) * np.log1p(-1.0 / d)) * np.power(d, alpha + 1.0)
        
    m0 = D**alpha * pdiff_a / alpha
    A1oD = D**alpha * pdiff_a1 / (alpha + 1.0)
    KL = np.concatenate(([0.0], A1oD - (d - 1.0) * m0))
    KR = np.concatenate(([0.0], d * m0 - A1oD))
    ga = sgamma(alpha)
    w_trapz = np.full(n_steps + 1, 1.0 / n_steps)
    w_trapz[0] *= 0.5
    w_trapz[-1] *= 0.5
    wM = (w_trapz[None, :] * Mmesh) / ga

    nf = 1 << int(np.ceil(np.log2(2 * (n_steps + 1))))
    wM_rev = wM[:, ::-1]
    rfft_KL = np.fft.rfft(KL, nf)
    rfft_KR = np.fft.rfft(KR, nf)

    conv_L = np.fft.irfft(np.fft.rfft(wM_rev, nf, axis=1) * rfft_KL[None, :], nf, axis=1)[:, :n_steps + 1]
    conv_R = np.fft.irfft(np.fft.rfft(wM_rev, nf, axis=1) * rfft_KR[None, :], nf, axis=1)[:, :n_steps + 1]

    AL = conv_L[:, n_steps:0:-1].T
    AR = conv_R[:, n_steps:0:-1].T

    T_tensor = (Mmesh[:, :-1].T[:, :, None] * AL[:, None, :] + 
                Mmesh[:, 1:].T[:, :, None]  * AR[:, None, :]) / D
    M_tens = T_tensor.reshape(n_steps, m1 * m1).T  # shape ((mhat+1)^2, n_steps)
    omega_inv = (2.0 * alpha * np.arange(m1) + 1.0)
    return M_tens, omega_inv

def build_S_fubini_batch(alpha: float, mhat: int, dB: np.ndarray, M_tens: np.ndarray | None = None) -> np.ndarray:
    """
    Evaluate S_alpha for an arbitrary batch of paths dB (shape (N_paths, n_steps))
    via a single BLAS matrix multiplication.
    """
    if dB.ndim == 1:
        dB = dB[None, :]
    n_paths, n_steps = dB.shape
    m1 = mhat + 1
    
    if M_tens is None:
        M_tens, omega_inv = build_fubini_tensor(alpha, mhat, n_steps)
    else:
        omega_inv = (2.0 * alpha * np.arange(m1) + 1.0)
        
    # Hardware-accelerated with PyTorch (Apple Accelerate / NEON / AMX)
    M_t = torch.from_numpy(M_tens)
    dB_t = torch.from_numpy(dB)
    Xi_t = torch.matmul(dB_t, M_t.T)
    Xi_all = Xi_t.numpy().reshape(n_paths, m1, m1)
    S_all = Xi_all * omega_inv[None, None, :]
    return S_all

def solve_affine_fubini_batch(
    alpha: float,
    mhat: int,
    dB: np.ndarray,
    y0: float,
    b0=0.0,
    b1=1.0,
    s0=0.0,
    s1=1.0,
    Nq: int = 64,
    lam_b: float = 1.0,
    lam_s: float = 1.0,
    t_eval: np.ndarray | None = None,
    M_tens: np.ndarray | None = None,
    trace_order: int = 1,
    correction: str = "operator_trace",
    tikhonov_reg: float = 1e-11,
    trace_quadrature_order: int = 32,
    kernel_quadrature_order: int = 32,
    bc_weight: float = 10.0,
) -> np.ndarray:
    """
    End-to-end fully vectorized & mathematically optimized affine solver for N_paths.
    Uses Stochastic Fubini tensor contraction + batched linear system solve.
    """
    if dB.ndim == 1:
        dB = dB[None, :]
    n_paths, n_steps = dB.shape
    m1 = mhat + 1
    
    # Process in chunks of max 10,000 paths to optimize memory locality and cache
    operator_mode = correction == "operator_trace"
    chunk_size = 64 if operator_mode else 10000
    if n_paths > chunk_size:
        if M_tens is None:
            M_tens, _ = build_fubini_tensor(alpha, mhat, n_steps)
        results = []
        for i in range(0, n_paths, chunk_size):
            chunk_dB = dB[i:i + chunk_size]
            res_chunk = solve_affine_fubini_batch(
                alpha=alpha,
                mhat=mhat,
                dB=chunk_dB,
                y0=y0,
                b0=b0,
                b1=b1,
                s0=s0,
                s1=s1,
                Nq=Nq,
                lam_b=lam_b,
                lam_s=lam_s,
                t_eval=t_eval,
                M_tens=M_tens,
                trace_order=trace_order,
                correction=correction,
                tikhonov_reg=tikhonov_reg,
                trace_quadrature_order=trace_quadrature_order,
                kernel_quadrature_order=kernel_quadrature_order,
                bc_weight=bc_weight,
            )
            results.append(res_chunk)
        return np.vstack(results)

    # 1. Compute S_all for all paths via Fubini tensor contraction
    S_all = build_S_fubini_batch(alpha, mhat, dB, M_tens)
    
    # 2. Collocation setup
    t_cheb = chebyshev_nodes(Nq)
    Phi = basis_eval(alpha, mhat, t_cheb)
    PhiT = Phi.T
    A = get_A(alpha, mhat)
    DetT = (t_cheb ** alpha)[:, None] * (PhiT @ A.T)
    
    Z = np.zeros((Nq, m1))
    b0v = b0(t_cheb) if callable(b0) else np.full(Nq, float(b0))
    s0v = s0(t_cheb) if callable(s0) else np.full(Nq, float(s0))
    wb, ws = np.sqrt(lam_b), np.sqrt(lam_s)
    
    # Legacy modes modify the local drift.  operator_trace instead inserts the
    # accumulated finite-dimensional trace in the OME block after the base solve.
    c0 = trace_coefficient(alpha)
    if correction in ("none", "operator_trace"):
        c_alpha_t = np.zeros(Nq)
    elif correction == "constant":
        c_alpha_t = np.full(Nq, c0)
    elif correction == "legacy_t_power":
        t_term0 = c0 * np.power(np.clip(t_cheb, 0.0, None), 2.0 * alpha - 1.0)
        if trace_order == 0:
            c_alpha_t = t_term0
        elif trace_order == 1:
            c1 = sgamma(3.0 * alpha - 1.0) / (2.0 * sgamma(alpha) * sgamma(2.0 * alpha))
            t_term1 = c1 * np.power(np.clip(t_cheb, 0.0, None), 3.0 * alpha - 1.0) * b1
            c_alpha_t = t_term0 + t_term1
        elif trace_order == 2:
            ml_series = np.zeros_like(t_cheb)
            z = b1 * np.power(np.clip(t_cheb, 0.0, None), alpha)
            for k in range(50):
                term = (z ** k) / sgamma(alpha * (k + 1) + 1.0)
                ml_series += term
                if np.max(np.abs(term)) < 1e-16:
                    break
            c_alpha_t = t_term0 * sgamma(alpha + 1.0) * ml_series
        else:
            c_alpha_t = t_term0
    else:
        raise ValueError(f"Unknown correction: {correction}")
        
    b0_eff = b0v - c_alpha_t * s0v * s1
    b1_eff = b1 - c_alpha_t * (s1 ** 2)
    
    R2 = wb * np.hstack([-b1_eff[:, None] * PhiT, PhiT, Z])
    R3 = ws * np.hstack([-s1 * PhiT, Z, PhiT])
    phi0 = basis_eval(alpha, mhat, np.array([0.0]))[:, 0]
    Rbc = bc_weight * np.concatenate([phi0, np.zeros(2 * m1)])[None, :]
    rhs = np.concatenate([np.full(Nq, y0), wb * b0_eff, ws * s0v, [bc_weight * y0]])
    
    # 3. Assemble batched linear system
    StoT_all = np.einsum("qk,rjk->rqj", PhiT, S_all)
    PhiT_all = np.broadcast_to(PhiT, (n_paths, Nq, m1))
    DetT_all = np.broadcast_to(DetT, (n_paths, Nq, m1))
    R1_all = np.concatenate([PhiT_all, -DetT_all, -StoT_all], axis=2)
    
    R2_all = np.broadcast_to(R2, (n_paths, Nq, 3 * m1))
    R3_all = np.broadcast_to(R3, (n_paths, Nq, 3 * m1))
    Rbc_all = np.broadcast_to(Rbc, (n_paths, 1, 3 * m1))
    Amat_all = np.concatenate([R1_all, R2_all, R3_all, Rbc_all], axis=1)
    
    # 4. Solve normal equations (Accelerated with PyTorch batched DGEMM + Apple Accelerate)
    Amat_t = torch.from_numpy(Amat_all)
    rhs_t = torch.from_numpy(rhs)
    
    AtA = torch.bmm(Amat_t.transpose(1, 2), Amat_t)
    reg_mat = tikhonov_reg * torch.eye(3 * m1, dtype=torch.float64)
    AtA_reg = AtA + reg_mat[None, :, :]
    Atb = torch.matmul(Amat_t.transpose(1, 2), rhs_t.unsqueeze(-1))
    
    if operator_mode:
        Hinv_t = torch.linalg.inv(AtA_reg)
        z_t = torch.matmul(Hinv_t, Atb).squeeze(-1)
    else:
        z_t = torch.linalg.solve(AtA_reg, Atb).squeeze(-1)

    if operator_mode and float(s1) != 0.0:
        prepared = prepare_operator_trace_quadrature(
            alpha=alpha,
            mhat=mhat,
            collocation_t=t_cheb,
            trace_t=t_cheb,
            trace_quadrature_order=trace_quadrature_order,
            kernel_quadrature_order=kernel_quadrature_order,
        )
        M_s, G, trace_weights, q_trace = prepared
        M_s_t = torch.from_numpy(M_s)
        G_t = torch.from_numpy(G)
        weights_t = torch.from_numpy(trace_weights)

        J_ome_t = Amat_t[:, :Nq, :]
        response_t = torch.bmm(Hinv_t, J_ome_t.transpose(1, 2))
        response_sigma_t = response_t[:, 2 * m1:3 * m1, :]
        hessian_sigma_t = Hinv_t[:, 2 * m1:3 * m1, 2 * m1:3 * m1]
        theta_sigma_t = z_t[:, 2 * m1:3 * m1]
        sigma_hat_t = torch.matmul(theta_sigma_t, M_s_t)

        base_rhs_t = rhs_t.unsqueeze(0).expand(n_paths, -1)
        residual_t = torch.bmm(Amat_t, z_t.unsqueeze(-1)).squeeze(-1) - base_rhs_t
        residual_ome_t = residual_t[:, :Nq]
        residual_contraction_t = torch.matmul(residual_ome_t, G_t)

        D_c_sigma_t = (
            torch.matmul(response_sigma_t, G_t) * sigma_hat_t[:, None, :]
            + torch.matmul(hessian_sigma_t, M_s_t)
            * residual_contraction_t[:, None, :]
        )
        integrand_t = torch.sum(D_c_sigma_t * M_s_t[None, :, :], dim=1)
        trace_t = torch.sum(
            integrand_t.reshape(n_paths, Nq, q_trace) * weights_t[None, :, :],
            dim=2,
        )

        corrected_rhs_t = base_rhs_t.clone()
        corrected_rhs_t[:, :Nq] -= trace_t
        corrected_Atb_t = torch.bmm(
            Amat_t.transpose(1, 2), corrected_rhs_t.unsqueeze(-1)
        )
        z_t = torch.matmul(Hinv_t, corrected_Atb_t).squeeze(-1)

    z_all = z_t.numpy()
    c_all = z_all[:, :m1]
    
    # 5. Evaluate on t_eval
    if t_eval is not None:
        Phi_eval = basis_eval(alpha, mhat, t_eval)  # (m1, N_eval)
        sols = c_all @ Phi_eval                     # (n_paths, N_eval)
        return sols
    return c_all


def solve_nonlinear_fubini_batch(
    alpha: float,
    mhat: int,
    dB: np.ndarray,
    y0: float,
    bfun,
    bprime,
    bprime2,
    sfun,
    sprime,
    sprime2,
    Nq: int = 64,
    lam_b: float = 1.0,
    lam_s: float = 1.0,
    t_eval: np.ndarray | None = None,
    M_tens: np.ndarray | None = None,
    correction: str = "operator_trace",
    max_iter: int = 30,
    tol: float = 1e-10,
    tikhonov_reg: float = 1e-10,
    trace_quadrature_order: int = 32,
    kernel_quadrature_order: int = 32,
    chunk_size: int = 32,
    bc_weight: float = 10.0,
    return_trace: bool = False,
):
    """Batched nonlinear collocation with the finite-dimensional operator trace.

    The callables accept ``(t, y)`` NumPy arrays.  The first solve is an uncorrected
    Gauss--Newton solve.  In ``operator_trace`` mode its exact normal-equation Hessian
    (including residual-weighted curvature) is differentiated, and a second solve is
    performed with that accumulated trace frozen in the OME residual.
    """
    dB = np.asarray(dB, dtype=float)
    if dB.ndim == 1:
        dB = dB[None, :]
    n_paths, n_steps = dB.shape
    if correction not in ("none", "operator_trace"):
        raise ValueError("nonlinear batch solver supports 'none' or 'operator_trace'")
    if correction == "operator_trace" and (bprime2 is None or sprime2 is None):
        raise ValueError("operator_trace requires bprime2 and sprime2")

    if n_paths > chunk_size:
        if M_tens is None:
            M_tens, _ = build_fubini_tensor(alpha, mhat, n_steps)
        solved, traces = [], []
        for start in range(0, n_paths, chunk_size):
            result = solve_nonlinear_fubini_batch(
                alpha, mhat, dB[start:start + chunk_size], y0,
                bfun, bprime, bprime2, sfun, sprime, sprime2,
                Nq=Nq, lam_b=lam_b, lam_s=lam_s, t_eval=t_eval,
                M_tens=M_tens, correction=correction, max_iter=max_iter,
                tol=tol, tikhonov_reg=tikhonov_reg,
                trace_quadrature_order=trace_quadrature_order,
                kernel_quadrature_order=kernel_quadrature_order,
                chunk_size=chunk_size, return_trace=return_trace,
                bc_weight=bc_weight,
            )
            if return_trace:
                values, trace = result
                solved.append(values)
                traces.append(trace)
            else:
                solved.append(result)
        values = np.vstack(solved)
        return (values, np.vstack(traces)) if return_trace else values

    m1 = mhat + 1
    S_all = build_S_fubini_batch(alpha, mhat, dB, M_tens)
    t = chebyshev_nodes(Nq)
    PhiT = basis_eval(alpha, mhat, t).T
    A = get_A(alpha, mhat)
    DetT = (t ** alpha)[:, None] * (PhiT @ A.T)
    StoT = np.einsum("qk,rjk->rqj", PhiT, S_all)
    wb, ws = np.sqrt(lam_b), np.sqrt(lam_s)
    zero = np.zeros((n_paths, Nq, m1))
    Phi = np.broadcast_to(PhiT, (n_paths, Nq, m1))
    Det = np.broadcast_to(DetT, (n_paths, Nq, m1))
    J1 = np.concatenate([Phi, -Det, -StoT], axis=2)
    phi0 = basis_eval(alpha, mhat, np.array([0.0]))[:, 0]
    Jbc = bc_weight * np.broadcast_to(
        np.concatenate([phi0, np.zeros(2 * m1)])[None, None, :],
        (n_paths, 1, 3 * m1),
    )

    def values(fun, state):
        out = np.asarray(fun(t[None, :], state), dtype=float)
        return np.broadcast_to(out, state.shape).copy()

    # Affine Taylor initialization about y0, without any trace feedback.
    state0 = np.full((n_paths, Nq), y0)
    b1 = values(bprime, state0)
    s1 = values(sprime, state0)
    b0 = values(bfun, state0) - b1 * y0
    s0 = values(sfun, state0) - s1 * y0
    J2 = wb * np.concatenate([-b1[..., None] * Phi, Phi, zero], axis=2)
    J3 = ws * np.concatenate([-s1[..., None] * Phi, zero, Phi], axis=2)
    matrix = np.concatenate([J1, J2, J3, Jbc], axis=1)
    rhs = np.concatenate([
        np.full((n_paths, Nq), y0), wb * b0, ws * s0,
        np.full((n_paths, 1), bc_weight * y0),
    ], axis=1)
    matrix_t = torch.from_numpy(matrix)
    rhs_t = torch.from_numpy(rhs)
    normal = torch.bmm(matrix_t.transpose(1, 2), matrix_t)
    eye = torch.eye(3 * m1, dtype=torch.float64)[None]
    normal_rhs = torch.bmm(matrix_t.transpose(1, 2), rhs_t.unsqueeze(-1))
    z = torch.linalg.solve(normal + tikhonov_reg * eye, normal_rhs).squeeze(-1).numpy()

    fixed_trace = np.zeros((n_paths, Nq))

    def residual_and_jacobian(z_now):
        c = z_now[:, :m1]
        tb = z_now[:, m1:2 * m1]
        ts = z_now[:, 2 * m1:]
        state = c @ PhiT.T
        b = values(bfun, state)
        bp = values(bprime, state)
        s = values(sfun, state)
        sp = values(sprime, state)
        r1 = state - y0 - tb @ DetT.T - np.einsum("rqj,rj->rq", StoT, ts) + fixed_trace
        r2 = wb * (tb @ PhiT.T - b)
        r3 = ws * (ts @ PhiT.T - s)
        rbc = bc_weight * (c @ phi0 - y0)[:, None]
        residual = np.concatenate([r1, r2, r3, rbc], axis=1)
        j2 = wb * np.concatenate([-bp[..., None] * Phi, Phi, zero], axis=2)
        j3 = ws * np.concatenate([-sp[..., None] * Phi, zero, Phi], axis=2)
        jacobian = np.concatenate([J1, j2, j3, Jbc], axis=1)
        return residual, jacobian, state

    def optimize(z_start):
        z_now = z_start.copy()
        for _ in range(max_iter):
            residual, jacobian, state = residual_and_jacobian(z_now)
            jac_t = torch.from_numpy(jacobian)
            res_t = torch.from_numpy(residual)
            jtj = torch.bmm(jac_t.transpose(1, 2), jac_t)
            jtr = torch.bmm(jac_t.transpose(1, 2), res_t.unsqueeze(-1))
            diag = torch.diag_embed(torch.clamp(torch.diagonal(jtj, dim1=1, dim2=2), min=1e-8))
            dz = torch.linalg.solve(jtj + 1e-8 * diag + tikhonov_reg * eye, -jtr).squeeze(-1).numpy()
            # A shared backtracking factor keeps all paths vectorized.
            old_norm = np.sum(residual * residual, axis=1)
            step = 1.0
            while step >= 2.0 ** -12:
                candidate = z_now + step * dz
                candidate_residual, _, _ = residual_and_jacobian(candidate)
                if np.all(np.sum(candidate_residual * candidate_residual, axis=1) <= old_norm * (1.0 + 1e-12)):
                    break
                step *= 0.5
            z_now = candidate
            if np.max(np.linalg.norm(step * dz, axis=1)) < tol * max(1.0, np.max(np.linalg.norm(z_now, axis=1))):
                break
        return z_now, *residual_and_jacobian(z_now)

    z, residual, jacobian, state = optimize(z)
    trace = np.zeros((n_paths, Nq))
    if correction == "operator_trace":
        jac_t = torch.from_numpy(jacobian)
        hessian = torch.bmm(jac_t.transpose(1, 2), jac_t).numpy()
        rb = residual[:, Nq:2 * Nq]
        rs = residual[:, 2 * Nq:3 * Nq]
        curvature = (
            -wb * rb * values(bprime2, state)
            -ws * rs * values(sprime2, state)
        )
        hessian[:, :m1, :m1] += np.einsum("qi,rq,qj->rij", PhiT, curvature, PhiT)
        hessian_inv = np.linalg.pinv(hessian, hermitian=True)

        M_s, G, trace_weights, q = prepare_operator_trace_quadrature(
            alpha, mhat, t, t, trace_quadrature_order, kernel_quadrature_order
        )
        response = np.einsum("rij,rqj->riq", hessian_inv, J1)
        response_sigma = response[:, 2 * m1:3 * m1]
        hessian_sigma = hessian_inv[:, 2 * m1:3 * m1, 2 * m1:3 * m1]
        sigma_hat = z[:, 2 * m1:3 * m1] @ M_s
        residual_contraction = residual[:, :Nq] @ G
        d_c_sigma = (
            np.einsum("riq,qs->ris", response_sigma, G) * sigma_hat[:, None, :]
            + np.einsum("rij,js->ris", hessian_sigma, M_s)
            * residual_contraction[:, None, :]
        )
        integrand = np.sum(d_c_sigma * M_s[None, :, :], axis=1)
        trace = np.sum(integrand.reshape(n_paths, Nq, q) * trace_weights[None], axis=2)
        fixed_trace[:] = trace
        z, residual, jacobian, state = optimize(z)

    result = z[:, :m1]
    if t_eval is not None:
        result = result @ basis_eval(alpha, mhat, np.asarray(t_eval, dtype=float))
    return (result, trace) if return_trace else result

def parallel_solve_affine(
    alpha: float,
    mhat: int,
    dB: np.ndarray,
    y0: float,
    b0=0.0,
    b1=1.0,
    s0=0.0,
    s1=1.0,
    Nq: int = 64,
    lam_b: float = 1.0,
    lam_s: float = 1.0,
    t_eval: np.ndarray | None = None,
    n_workers: int | None = None,
    trace_order: int = 1,
    correction: str = "operator_trace",
    tikhonov_reg: float = 1e-11,
    trace_quadrature_order: int = 32,
    kernel_quadrature_order: int = 32,
    bc_weight: float = 10.0,
) -> np.ndarray:
    """Wrapper that routes to the ultra-fast Stochastic Fubini batched solver."""
    return solve_affine_fubini_batch(
        alpha=alpha,
        mhat=mhat,
        dB=dB,
        y0=y0,
        b0=b0,
        b1=b1,
        s0=s0,
        s1=s1,
        Nq=Nq,
        lam_b=lam_b,
        lam_s=lam_s,
        t_eval=t_eval,
        trace_order=trace_order,
        correction=correction,
        tikhonov_reg=tikhonov_reg,
        trace_quadrature_order=trace_quadrature_order,
        kernel_quadrature_order=kernel_quadrature_order,
        bc_weight=bc_weight,
    )

def _em_caputo_chunk(args):
    chunk_dB, alpha, bfun, sfun, y0, t_eval = args
    R, n = chunk_dB.shape
    D = 1.0 / n
    tm = np.linspace(0.0, 1.0, n + 1)
    d = np.arange(1, n + 1, dtype=float)
    with np.errstate(divide="ignore"):
        pdiff_a = -np.expm1(alpha * np.log1p(-1.0 / d)) * np.power(d, alpha)
    wdet = D**alpha * pdiff_a / alpha
    ksto = np.power(d * D, alpha - 1.0)
    ga = sgamma(alpha)
    
    y = np.empty((R, n + 1))
    y[:, 0] = y0
    Bh = np.empty((R, n))
    Sh = np.empty((R, n))
    
    for k in range(n):
        Bh[:, k] = bfun(tm[k], y[:, k])
        Sh[:, k] = sfun(tm[k], y[:, k]) * chunk_dB[:, k]
        wd = wdet[k::-1]
        ks = ksto[k::-1]
        y[:, k + 1] = y0 + (Bh[:, :k + 1] @ wd + Sh[:, :k + 1] @ ks) / ga
        
    if t_eval is not None:
        return np.array([np.interp(t_eval, tm, y[i]) for i in range(R)])
    return y

def parallel_em_caputo(
    alpha: float,
    bfun,
    sfun,
    y0: float,
    dB: np.ndarray,
    t_eval: np.ndarray | None = None,
    n_workers: int | None = None
) -> np.ndarray:
    """Parallelized Fractional Euler-Maruyama solver across Monte Carlo paths."""
    if dB.ndim == 1:
        dB = dB[None, :]
    n_paths, n_steps = dB.shape
    
    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, n_paths)
        
    chunks = np.array_split(dB, n_workers)
    worker_args = [(chunk, alpha, bfun, sfun, y0, t_eval) for chunk in chunks if len(chunk) > 0]
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results_nested = list(executor.map(_em_caputo_chunk, worker_args))
        
    return np.vstack(results_nested)
