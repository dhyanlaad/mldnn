# Stochastic Fractional Ornstein–Uhlenbeck Process: Mean Error Analysis

This document contains the analytical vs numerical mean error tables for the stochastic fractional Ornstein–Uhlenbeck (fOU) process across various fractional orders $\alpha$ and spectral truncation degrees $\hat{m}$.

---

## 1. Problem Formulation & Exact Analytic Mean

The stochastic fractional Ornstein–Uhlenbeck (fOU) process is defined by:

$$D_t^\alpha X(t) = -\theta X(t) + \sigma \frac{\mathrm{d}W_t}{\mathrm{d}t}, \quad t \in [0, 1], \quad X(0) = X_0$$

### Model Parameters:
- **Mean-reversion rate**: $\theta = 0.3$
- **Long-term mean**: $\mu_{\text{OU}} = 0.0$
- **Volatility**: $\sigma = 0.15$
- **Initial condition**: $X_0 = 1.0$
- **Fractional orders**: $\alpha \in \{0.6, 0.7, 0.8, 0.9, 1.0\}$
- **Spectral truncation degrees**: $\hat{m} \in \{2, 4, 8, 16, 24, 32\}$
- **Brownian mesh**: $N = 65{,}536$ continuous time-steps
- **Monte Carlo sample paths**: $R = 5{,}000$ independent trajectories
- **Evaluation grid**: $n = 101$ equidistant nodes in $t \in [0, 1]$ ($t_k = \frac{k-1}{100}, k = 1, \dots, 101$)

### Exact Analytical Mean:
Taking expectations on both sides yields the deterministic linear Caputo fractional differential equation $D_t^\alpha \mathbb{E}[X(t)] = -\theta \mathbb{E}[X(t)]$ with $\mathbb{E}[X(0)] = X_0$, whose analytical solution is:

$$\mu(t) \equiv \mathbb{E}[X(t)] = X_0 E_\alpha(-\theta t^\alpha) = E_\alpha(-0.3 t^\alpha) = \sum_{k=0}^\infty \frac{(-0.3 t^\alpha)^k}{\Gamma(k\alpha + 1)}$$

### Numerical Empirical Mean:
$$\bar{X}(t_k) = \frac{1}{R} \sum_{p=1}^R X_{\mathrm{MLDNN}}^{(p)}(t_k)$$

---

## 2. Discrete $L_2$ Error Table

$$\text{Discrete } L_2 \text{ Error} = \frac{1}{n} \left( \sum_{k=1}^n \left( \bar{X}(t_k) - \mu(t_k) \right)^2 \right)^{1/2}$$

| $\hat{m}$ | $\alpha = 0.6$ | $\alpha = 0.7$ | $\alpha = 0.8$ | $\alpha = 0.9$ | $\alpha = 1.0$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$2$**  | $1.9104 \times 10^{-4}$ | $1.7526 \times 10^{-4}$ | $1.6294 \times 10^{-4}$ | $1.5361 \times 10^{-4}$ | $1.4665 \times 10^{-4}$ |
| **$4$**  | $2.0364 \times 10^{-4}$ | $1.8535 \times 10^{-4}$ | $1.7116 \times 10^{-4}$ | $1.6032 \times 10^{-4}$ | $1.5207 \times 10^{-4}$ |
| **$8$**  | $2.1159 \times 10^{-4}$ | $1.8963 \times 10^{-4}$ | $1.7342 \times 10^{-4}$ | $1.6155 \times 10^{-4}$ | $1.5274 \times 10^{-4}$ |
| **$16$** | $2.1838 \times 10^{-4}$ | $1.9428 \times 10^{-4}$ | $1.7540 \times 10^{-4}$ | $1.6262 \times 10^{-4}$ | $1.5331 \times 10^{-4}$ |
| **$24$** | $2.2088 \times 10^{-4}$ | $1.9397 \times 10^{-4}$ | $1.7554 \times 10^{-4}$ | $1.6285 \times 10^{-4}$ | $1.5346 \times 10^{-4}$ |
| **$32$** | $2.2000 \times 10^{-4}$ | $1.9457 \times 10^{-4}$ | $1.7610 \times 10^{-4}$ | $1.6302 \times 10^{-4}$ | $1.5356 \times 10^{-4}$ |

---

## 3. Discrete $L_\infty$ Error Table

$$\text{Discrete } L_\infty \text{ Error} = \max_{1 \le k \le n} \left| \bar{X}(t_k) - \mu(t_k) \right|$$

| $\hat{m}$ | $\alpha = 0.6$ | $\alpha = 0.7$ | $\alpha = 0.8$ | $\alpha = 0.9$ | $\alpha = 1.0$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$2$**  | $3.3628 \times 10^{-3}$ | $2.8247 \times 10^{-3}$ | $2.4521 \times 10^{-3}$ | $2.1931 \times 10^{-3}$ | $2.0153 \times 10^{-3}$ |
| **$4$**  | $3.6697 \times 10^{-3}$ | $3.2633 \times 10^{-3}$ | $2.9305 \times 10^{-3}$ | $2.6502 \times 10^{-3}$ | $2.4114 \times 10^{-3}$ |
| **$8$**  | $4.1382 \times 10^{-3}$ | $3.7675 \times 10^{-3}$ | $3.3801 \times 10^{-3}$ | $3.0077 \times 10^{-3}$ | $2.6821 \times 10^{-3}$ |
| **$16$** | $4.8334 \times 10^{-3}$ | $4.0674 \times 10^{-3}$ | $3.5007 \times 10^{-3}$ | $3.0335 \times 10^{-3}$ | $2.6184 \times 10^{-3}$ |
| **$24$** | $4.7652 \times 10^{-3}$ | $4.0875 \times 10^{-3}$ | $3.5206 \times 10^{-3}$ | $3.0398 \times 10^{-3}$ | $2.6179 \times 10^{-3}$ |
| **$32$** | $5.2964 \times 10^{-3}$ | $4.5255 \times 10^{-3}$ | $3.7901 \times 10^{-3}$ | $3.1816 \times 10^{-3}$ | $2.6614 \times 10^{-3}$ |

---

## 4. Associated Data Artifacts
- Compressed Data Cache: [`data_cache.npz`](data_cache.npz)
- Discrete $L_2$ Error CSV: [`discrete_l2_mean_error.csv`](discrete_l2_mean_error.csv)
- Discrete $L_\infty$ Error CSV: [`discrete_linf_mean_error.csv`](discrete_linf_mean_error.csv)
