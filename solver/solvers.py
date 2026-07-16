import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .core_mldnn import get_A, build_S, basis_eval, chebyshev_nodes, em_caputo

torch.set_default_dtype(torch.float64)

class FEMSolver:
    def __init__(self, alpha, bfun, sfun):
        self.alpha = alpha
        self.bfun = bfun
        self.sfun = sfun

    def solve(self, y0, dB):
        return em_caputo(self.alpha, self.bfun, self.sfun, y0, dB)

class MLDNNNetwork(nn.Module):
    def __init__(self, mhat, hidden_dim=32, num_layers=3):
        super().__init__()
        self.mhat = mhat
        layers = []
        layers.append(nn.Linear(mhat + 1, hidden_dim))
        layers.append(nn.Tanh())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        
    def forward(self, M):
        return self.net(M).squeeze(-1)

class MLDNNSolver:
    def __init__(self, alpha, mhat, bfun, sfun, y0, S, Nq=64):
        self.alpha = alpha
        self.mhat = mhat
        self.bfun = bfun
        self.sfun = sfun
        self.y0 = y0
        self.Nq = Nq
        self.S = S
        self.device = torch.device('cpu') # Use CPU for robust float64 operations
        
        self.model = MLDNNNetwork(mhat).to(self.device)
        self.theta_b = nn.Parameter(torch.zeros(mhat + 1, device=self.device))
        self.theta_s = nn.Parameter(torch.zeros(mhat + 1, device=self.device))
        
        self.t = chebyshev_nodes(Nq)
        self.M_t = basis_eval(alpha, mhat, self.t).T
        
        A = get_A(alpha, mhat)
        
        DetT = (self.t ** alpha)[:, None] * (self.M_t @ A.T)
        StoT = self.M_t @ S.T if S is not None else np.zeros_like(DetT)
        
        self.M_t_ts = torch.tensor(self.M_t, dtype=torch.float64, device=self.device)
        self.DetT_ts = torch.tensor(DetT, dtype=torch.float64, device=self.device)
        self.StoT_ts = torch.tensor(StoT, dtype=torch.float64, device=self.device)
        self.t_ts = torch.tensor(self.t, dtype=torch.float64, device=self.device)

    def train(self, epochs=50, lr=0.1, lam_b=1.0, lam_s=1.0):
        # We use LBFGS as it is perfectly suited for this formulation and reaches machine precision
        optimizer = optim.LBFGS(list(self.model.parameters()) + [self.theta_b, self.theta_s], 
                                lr=1.0, max_iter=epochs, tolerance_grad=1e-13, tolerance_change=1e-13,
                                line_search_fn="strong_wolfe")
        
        def closure():
            optimizer.zero_grad()
            N_t = self.model(self.M_t_ts)
            
            b_ts = self.bfun(self.t_ts, N_t)
            s_ts = self.sfun(self.t_ts, N_t)
            
            loss_b = torch.mean((self.M_t_ts @ self.theta_b - b_ts)**2)
            loss_s = torch.mean((self.M_t_ts @ self.theta_s - s_ts)**2)
            
            int_b = self.DetT_ts @ self.theta_b
            int_s = self.StoT_ts @ self.theta_s
            
            loss_sde = torch.mean((N_t - self.y0 - int_b - int_s)**2)
            
            loss = loss_sde + lam_b * loss_b + lam_s * loss_s
            loss.backward()
            return loss
            
        optimizer.step(closure)
        
    def evaluate(self, t):
        t_np = np.asarray(t, dtype=np.float64)
        M_t = basis_eval(self.alpha, self.mhat, t_np).T
        M_t_ts = torch.tensor(M_t, dtype=torch.float64, device=self.device)
        with torch.no_grad():
            N_t = self.model(M_t_ts).cpu().numpy()
        return N_t
