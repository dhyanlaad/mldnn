# Malliavin Trace Correction: A Beginner-Friendly Explanation

## 1. Why is a correction needed?

We want to approximate a stochastic Volterra integral equation (SVIE) of the form

$$
y(t)
= y_0
+ \int_0^t K_\alpha(t,s)b(s,y(s))\,ds
+ \int_0^t K_\alpha(t,s)\sigma(s,y(s))\,dB_s.
$$

Here:

- $t\in[0,1]$ is time.
- $y(t)$ is the unknown random process that we want to compute.
- $y_0$ is its initial value.
- $b(t,y)$ is the **drift coefficient**. It describes the ordinary, non-random part of the dynamics.
- $\sigma(t,y)$ is the **diffusion coefficient**. It controls the strength of the random forcing.
- $B_t$ is standard Brownian motion.
- $dB_s$ indicates stochastic integration with respect to Brownian motion.
- $K_\alpha(t,s)$ is a fractional memory kernel:

  $$
  K_\alpha(t,s)
  =\frac{\mathbf 1_{\{0\leq s<t\}}}{\Gamma(\alpha)}(t-s)^{\alpha-1}.
  $$

- $\alpha\in(1/2,1]$ controls the strength of the memory. When $\alpha=1$, the kernel is simply $1$ for $s<t$, and the equation becomes an ordinary stochastic differential equation written in integral form.
- $\Gamma(\alpha)$ is the Gamma function, which extends the factorial: $\Gamma(k)=(k-1)!$ for positive integers $k$.
- $\mathbf 1_{\{0\leq s<t\}}$ is an indicator: it equals $1$ when $0\leq s<t$ and $0$ otherwise.

The final integral is intended to be an **Itô integral**. An Itô integrand at time $s$ may depend on the Brownian path up to time $s$, but it may not look into the future. Such a process is called **adapted**.

The numerical method uses one global polynomial expansion over the full time interval. Its fitted coefficients are obtained by solving one least-squares problem using all collocation times at once. Therefore, a coefficient fitted at an early time can depend on Brownian values at later times. The fitted coefficient is then generally **non-adapted**, or **anticipative**.

That distinction matters. Multiplying a Brownian operational matrix by a random, non-adapted coefficient does not automatically produce the desired Itô integral. The difference is an explicit term called the **Malliavin trace correction**.

The main result is:

$$
\boxed{
\text{corrected stochastic term}
=\text{algebraic stochastic term}
-\text{Malliavin trace}
}
$$

There is also a separate finite-basis projection error. The trace correction does not remove that error.

---

## 2. Numerical notation

Let

$$
\mathbf M(t)
=\begin{bmatrix}M_0^\Lambda(t)&\cdots&M_n^\Lambda(t)\end{bmatrix}^{\!\top}
\in\mathbb R^m,
\qquad m=n+1,
$$

where $M_0^\Lambda,\ldots,M_n^\Lambda$ are the chosen Müntz--Legendre basis functions. The superscript $\Lambda$ identifies the exponent sequence used to construct this basis. The precise polynomial formulas are not needed for the correction; what matters is that $\mathbf M(t)$ is deterministic and has $m$ entries.

The numerical solution and the fitted drift and diffusion are expanded as

$$
y_n(t)=\mathbf c^\top\mathbf M(t),
\qquad
b_n(t)=\mathbf c_b^\top\mathbf M(t),
\qquad
\sigma_n(t)=\mathbf c_\sigma^\top\mathbf M(t).
$$

The vectors $\mathbf c$, $\mathbf c_b$, and $\mathbf c_\sigma$ contain the fitted basis coefficients. The superscript $\top$ means transpose, so $\mathbf c^\top\mathbf M(t)$ is an inner product.

The method uses:

- $\mathbf P_\alpha(t)$: a deterministic operational object representing convolution with the ordinary fractional kernel;
- $\mathbf S_\alpha$: a random stochastic operational matrix representing convolution against Brownian motion;
- $T_c=\{t_1,\ldots,t_{N_c}\}$: the collocation times at which the equation is enforced;
- $N_c$: the number of collocation points;
- $n$: the largest basis index, so the basis has $m=n+1$ functions.

Before correction, the numerical stochastic term at time $t$ is

$$
\mathcal A_n(t)
=(\mathbf c_\sigma^*)^\top\mathbf S_\alpha\mathbf M(t).
$$

The star in $\mathbf c_\sigma^*$ means that this is the coefficient vector at the least-squares optimum.

---

## 3. The minimum Malliavin calculus we need

Malliavin calculus can be viewed as differentiation with respect to the driving Brownian path.

### 3.1 Brownian motion as a random input

Let

$$
H=L^2(0,1).
$$

This is the space of square-integrable deterministic functions on $(0,1)$. Its inner product and norm are

$$
\langle f,g\rangle_H=\int_0^1 f(s)g(s)\,ds,
\qquad
\|f\|_H^2=\int_0^1|f(s)|^2\,ds.
$$

For $h\in H$, write

$$
B(h)=\int_0^1h(s)\,dB_s.
$$

This is a Wiener integral: a Gaussian random variable made by integrating a deterministic function against Brownian motion.

### 3.2 The Malliavin derivative

Consider a random variable that depends smoothly on finitely many Wiener integrals:

$$
F=f\bigl(B(h_1),\ldots,B(h_k)\bigr).
$$

Its Malliavin derivative at Brownian time $s$ is

$$
D_sF
=\sum_{j=1}^k
\frac{\partial f}{\partial x_j}
\bigl(B(h_1),\ldots,B(h_k)\bigr)h_j(s).
$$

Informally, $D_sF$ answers:

> How sensitive is $F$ to a very small change in the Brownian path near time $s$?

For example,

$$
D_sB(h)=h(s).
$$

The space $\mathbb D^{1,2}$ contains square-integrable random variables whose Malliavin derivatives are also square-integrable. Its norm is

$$
\|F\|_{\mathbb D^{1,2}}^2
=\mathbb E[F^2]
+\mathbb E\!\left[\int_0^1|D_sF|^2\,ds\right].
$$

Here $\mathbb E$ denotes expectation, or averaging over all Brownian paths.

### 3.3 The divergence or Skorokhod integral

The operator $\delta$ is the adjoint of the Malliavin derivative. It is defined through

$$
\mathbb E[F\,\delta(u)]
=\mathbb E[\langle DF,u\rangle_H].
$$

The quantity $\delta(u)$ is called the **divergence integral** or **Skorokhod integral** of $u$. It allows certain non-adapted integrands, unlike the Itô integral.

If $u_s$ is adapted and sufficiently square-integrable, then

$$
\delta(u)=\int_0^1u_s\,dB_s
$$

is the usual Itô integral.

The formula that creates the trace correction is the divergence multiplication rule:

$$
F\,\delta(u)
=\delta(Fu)+\langle DF,u\rangle_H.
$$

Equivalently,

$$
\delta(Fu)
=F\,\delta(u)-\langle DF,u\rangle_H.
$$

If $F$ is deterministic, then $DF=0$, so there is no correction. If $F$ depends on the Brownian path, the inner-product term must be included.

### 3.4 Why $\alpha>1/2$ appears

For fixed $t$, the kernel must be square-integrable in $s$:

$$
\int_0^t K_\alpha(t,s)^2\,ds<\infty.
$$

Near $s=t$, its square behaves like $(t-s)^{2\alpha-2}$. This is integrable exactly when $2\alpha-2>-1$, or $\alpha>1/2$. This condition ensures that the kernels used in the divergence integral belong to $H=L^2(0,1)$.

---

## 4. Differentiating the stochastic operational matrix

The stochastic matrix is built as

$$
\mathbf S_\alpha=\boldsymbol\Xi\boldsymbol\Omega^{-1}.
$$

Here:

- $\boldsymbol\Omega$ is a deterministic Gram matrix of basis inner products;
- $\boldsymbol\Xi$ is a random cross-covariance matrix built from Wiener integrals;
- $\boldsymbol\Omega^{-1}$ is the matrix inverse of $\boldsymbol\Omega$.

Define

$$
\mathbf k^{(\alpha)}(s)
=\begin{bmatrix}
K_0^{(\alpha)}(s)&\cdots&K_n^{(\alpha)}(s)
\end{bmatrix}^{\!\top},
$$

where each $K_j^{(\alpha)}$ is a deterministic kernel produced by the stochastic Fubini reduction used to construct $\boldsymbol\Xi$.

Each matrix entry has the form

$$
\Xi_{ij}
=\int_0^1K_j^{(\alpha)}(s)M_i^\Lambda(s)\,dB_s.
$$

Because the integrand is deterministic,

$$
D_s\Xi_{ij}
=K_j^{(\alpha)}(s)M_i^\Lambda(s).
$$

In matrix form,

$$
D_s\boldsymbol\Xi
=\mathbf M(s)\mathbf k^{(\alpha)}(s)^\top,
$$

and therefore

$$
\boxed{
D_s\mathbf S_\alpha
=\mathbf M(s)\mathbf k^{(\alpha)}(s)^\top\boldsymbol\Omega^{-1}.
}
$$

This is an **outer product**: a column vector multiplied by a row vector. It has rank at most one. That low-rank structure is important computationally because it makes the sensitivity calculation much cheaper than differentiating every matrix entry independently.

### 4.1 The collocation matrix

Let $\boldsymbol\Phi_S$ be the matrix whose $i$th row is built from $\mathbf S_\alpha\mathbf M(t_i)$. Define

$$
g_i(s)
=\left\langle
\mathbf k^{(\alpha)}(s),
\boldsymbol\Omega^{-1}\mathbf M(t_i)
\right\rangle,
$$

and collect these values into

$$
\mathbf g(s)
=\begin{bmatrix}g_1(s)&\cdots&g_{N_c}(s)\end{bmatrix}^{\!\top}.
$$

Then

$$
\boxed{
D_s\boldsymbol\Phi_S
=\mathbf g(s)\mathbf M(s)^\top.
}
$$

Again, this is rank one. Both $\mathbf g(s)$ and $\mathbf M(s)$ are deterministic. The Brownian sensitivity entering the optimizer can therefore be assembled from deterministic quantities once the basis is fixed.

---

## 5. How the fitted coefficients depend on Brownian motion

Collect all trainable coefficients into one vector:

$$
\boldsymbol\theta
=\begin{bmatrix}\mathbf c\\\mathbf c_b\\\mathbf c_\sigma\end{bmatrix}.
$$

Let $\mathbf r(\boldsymbol\theta,\mathbf S_\alpha)$ be the residual vector. Its entries measure how far the numerical approximation is from satisfying:

1. the stochastic Volterra equation at the collocation points;
2. the drift-projection relation;
3. the diffusion-projection relation.

For one fixed Brownian path, the stochastic matrix $\mathbf S_\alpha$ is known data, and the least-squares objective is

$$
L(\boldsymbol\theta;\mathbf S_\alpha)
=\frac12\|\mathbf r(\boldsymbol\theta,\mathbf S_\alpha)\|_2^2
=\frac12\sum_{\ell=1}^q r_\ell(\boldsymbol\theta,\mathbf S_\alpha)^2.
$$

The factor $1/2$ only simplifies derivatives; it does not change the minimizer. If there are $q$ residual entries and $p$ fitted coefficients, the residual Jacobian is the $q\times p$ matrix

$$
\mathbf J
=\frac{\partial\mathbf r}{\partial\boldsymbol\theta}.
$$

Its entry $J_{\ell j}=\partial r_\ell/\partial\theta_j$ measures how residual $r_\ell$ changes when coefficient $\theta_j$ changes. The chain rule gives

$$
\nabla_{\boldsymbol\theta}L
=\sum_{\ell=1}^q r_\ell\nabla_{\boldsymbol\theta}r_\ell
=\mathbf J^\top\mathbf r.
$$

The text gives this gradient its own name:

$$
\boxed{
\mathbf F(\boldsymbol\theta,\mathbf S_\alpha)
=\mathbf J(\boldsymbol\theta,\mathbf S_\alpha)^\top
\mathbf r(\boldsymbol\theta,\mathbf S_\alpha)
=\nabla_{\boldsymbol\theta}L.
}
$$

Thus, $\mathbf F$ is not a new physical force or stochastic process. It is the vector of slopes of the fitting loss with respect to all coefficients. It is called the **normal-equation map** because setting it to zero produces the nonlinear least-squares normal equations.

At an unconstrained interior minimizer $\boldsymbol\theta^*$, every sufficiently small coefficient perturbation must have zero first-order effect on the loss. Therefore, the **first-order optimality condition** is

$$
\boxed{
\mathbf F(\boldsymbol\theta^*,\mathbf S_\alpha)
=\mathbf J(\boldsymbol\theta^*,\mathbf S_\alpha)^\top
\mathbf r(\boldsymbol\theta^*,\mathbf S_\alpha)
=\mathbf0.
}
$$

Geometrically, this says that the remaining residual is perpendicular to every local direction in which changing the coefficients can move the fitted model. It does **not** require $\mathbf r=\mathbf0$: a finite basis may be unable to satisfy every equation exactly, even at its best fit.

This zero-gradient condition is necessary for an interior local minimum, but it is not sufficient by itself: a maximum or saddle point can also have zero gradient. Positive curvature of the loss, described by the Hessian below, is what distinguishes a strict local minimum. If the coefficients are constrained and the solution lies on a constraint boundary, the appropriate Karush--Kuhn--Tucker conditions replace the simple equation $\mathbf F=\mathbf0$.

For comparison, if the residual is linear,

$$
\mathbf r=\mathbf K\boldsymbol\theta-\mathbf d,
$$

then $\mathbf J=\mathbf K$, and the condition becomes the familiar linear normal equation

$$
\mathbf K^\top(\mathbf K\boldsymbol\theta^*-\mathbf d)=\mathbf0,
\qquad\text{or}\qquad
\mathbf K^\top\mathbf K\boldsymbol\theta^*=\mathbf K^\top\mathbf d.
$$

### 5.1 Assumptions needed for differentiation

The following regularity conditions are needed:

- The drift $b$ and diffusion $\sigma$ are sufficiently smooth in their state variable, so ordinary and Malliavin chain rules are valid.
- The relevant random variables and derivatives have finite second moments.
- The optimizer is an interior solution, rather than a point fixed by an active constraint.
- The Hessian-like matrix below is invertible and is not pathologically ill-conditioned.
- The optimizer chosen by the numerical algorithm is locally a smooth function of $\mathbf S_\alpha$.

These assumptions are important. Convergence of a Gauss--Newton iteration by itself does not prove Malliavin differentiability of the fitted coefficients.

### 5.2 Implicit differentiation of the normal equations

Differentiate

$$
\mathbf F(\boldsymbol\theta^*,\mathbf S_\alpha)=\mathbf 0
$$

with respect to the Brownian path at time $s$. This gives

$$
\boxed{
D_s\boldsymbol\theta^*
=-\mathbf H^{-1}D_s^{\mathrm{exp}}\mathbf F.
}
$$

The notation $D_s^{\mathrm{exp}}$ means: differentiate only the explicit Brownian dependence through $\mathbf S_\alpha$, while temporarily holding $\boldsymbol\theta$ fixed.

The matrix

$$
\mathbf H
=\mathbf J^\top\mathbf J
+\sum_\ell r_\ell\nabla_{\boldsymbol\theta}^2r_\ell
$$

is the exact Hessian of one half of the squared residual norm. Here:

- $r_\ell$ is the $\ell$th residual entry;
- $\nabla_{\boldsymbol\theta}^2r_\ell$ is the Hessian matrix of that residual entry;
- $\mathbf J^\top\mathbf J$ is the Gauss--Newton part of the Hessian.

The explicit forcing is

$$
D_s^{\mathrm{exp}}\mathbf F
=(D_s^{\mathrm{exp}}\mathbf J)^\top\mathbf r
+\mathbf J^\top D_s^{\mathrm{exp}}\mathbf r.
$$

Only the stochastic-equation residual block depends explicitly on $\mathbf S_\alpha$. Consequently,

$$
D_s^{\mathrm{exp}}\mathbf r
=\begin{bmatrix}
-(D_s\boldsymbol\Phi_S)\mathbf c_\sigma^*\\
\mathbf 0\\
\mathbf 0
\end{bmatrix}.
$$

The explicit derivative of $\mathbf J$ is also zero except in the block obtained by differentiating $-\boldsymbol\Phi_S$ with respect to $\mathbf c_\sigma$.

Finally, let

$$
\mathbf E_\sigma
=\begin{bmatrix}\mathbf 0&\mathbf 0&\mathbf I_m\end{bmatrix}
$$

be the **selection matrix** that extracts the diffusion coefficients from $\boldsymbol\theta$. Then

$$
\boxed{
D_s\mathbf c_\sigma^*
=\mathbf E_\sigma D_s\boldsymbol\theta^*.
}
$$

This vector tells us how the fitted diffusion coefficients respond to Brownian noise at time $s$. It is the quantity needed in the trace.

### 5.3 Why the exact Hessian matters

If every residual is affine in $\boldsymbol\theta$, then $\nabla^2r_\ell=0$, and

$$
\mathbf H=\mathbf J^\top\mathbf J.
$$

For a nonlinear model, the second term in $\mathbf H$ generally does not vanish. Replacing $\mathbf H$ by $\mathbf J^\top\mathbf J$ gives a Gauss--Newton approximation. Its error is proportional to the remaining residual, so this replacement is safest only when the fitted residual is demonstrably very small.

---

## 6. Closed form for an affine model

Suppose

$$
b(t,y)=b_0(t)+b_1y,
\qquad
\sigma(t,y)=\sigma_0(t)+\sigma_1y.
$$

This is called **affine** because each coefficient is a constant-plus-linear function of $y$. The least-squares residual can then be written

$$
\mathbf r=\mathbf K\boldsymbol\theta-\mathbf d,
$$

where $\mathbf K$ is the block design matrix and $\mathbf d$ is the deterministic right-hand side. If $\mathbf K$ has full column rank, then

$$
D_s\boldsymbol\theta^*
=-\mathbf K^+(D_s\mathbf K)\boldsymbol\theta^*
-(\mathbf K^\top\mathbf K)^{-1}(D_s\mathbf K)^\top\mathbf r^*.
$$

Here:

- $\mathbf K^+=(\mathbf K^\top\mathbf K)^{-1}\mathbf K^\top$ is the Moore--Penrose pseudoinverse when $\mathbf K$ has full column rank;
- $\mathbf r^*=\mathbf K\boldsymbol\theta^*-\mathbf d$ is the residual at the optimum;
- $D_s\mathbf K$ is zero except for the block containing $-\boldsymbol\Phi_S$, where

  $$
  D_s\mathbf K=-D_s\boldsymbol\Phi_S
  =-\mathbf g(s)\mathbf M(s)^\top.
  $$

The last term involving $\mathbf r^*$ must not be discarded unless the fitted system is actually consistent, meaning $\mathbf r^*=\mathbf 0$. With finite regularization or projection weights, the residual is generally nonzero.

---

## 7. Definition of the Malliavin trace

For a fixed evaluation time $t$, define the deterministic vector-valued kernel

$$
\mathbf u_t(s)
=K_\alpha(t,s)\mathbf M(s).
$$

Its components belong to $H$ because $\alpha>1/2$. Since $\mathbf u_t$ is deterministic, its componentwise divergence integral produces the vector of stochastic basis convolutions, denoted by $\mathbf I_n(t)$.

The operational matrix approximates this vector by projection:

$$
\mathbf S_\alpha\mathbf M(t)
=\mathbf I_n(t)-\boldsymbol\varepsilon_n(t).
$$

The vector $\boldsymbol\varepsilon_n(t)$ is the **stochastic projection remainder**: the part of $\mathbf I_n(t)$ not represented by the truncated basis.

The discrete Malliavin trace is

$$
\boxed{
\operatorname{Trace}_n(t)
=\int_0^t
K_\alpha(t,s)
(D_s\mathbf c_\sigma^*)^\top\mathbf M(s)\,ds.
}
$$

Interpretation:

- $D_s\mathbf c_\sigma^*$ measures how the fitted diffusion coefficients react to Brownian noise at time $s$;
- $\mathbf M(s)$ converts that coefficient sensitivity into the sensitivity of the fitted diffusion function;
- $K_\alpha(t,s)$ transports that effect from $s$ to the evaluation time $t$;
- the integral accumulates this influence over all past times $0\leq s<t$.

---

## 8. The exact finite-dimensional identity

Apply the divergence multiplication rule component by component to $(\mathbf c_\sigma^*)^\top\delta(\mathbf u_t)$. The result is

$$
(\mathbf c_\sigma^*)^\top\delta(\mathbf u_t)
=\delta\!\left((\mathbf c_\sigma^*)^\top\mathbf u_t\right)
+\operatorname{Trace}_n(t).
$$

Combining this with the projection relation gives

$$
\boxed{
\mathcal A_n(t)-\operatorname{Trace}_n(t)
=\delta\!\left(
K_\alpha(t,\cdot)(\mathbf c_\sigma^*)^\top\mathbf M(\cdot)
\right)
-(\mathbf c_\sigma^*)^\top\boldsymbol\varepsilon_n(t).
}
$$

This identity is exact at a fixed finite basis size. It establishes three points:

1. The trace has a **minus sign** in the corrected algebraic term.
2. After correction, the main term is the Skorokhod integral of the fitted diffusion.
3. The separate projection error $(\mathbf c_\sigma^*)^\top\boldsymbol\varepsilon_n(t)$ remains.

It is tempting to call the uncorrected product “Stratonovich-like,” but that wording is only shorthand. The algebraic identity alone does not prove convergence to a particular pathwise or symmetric Stratonovich integral. Such a claim would require an additional trace-limit theorem.

---

## 9. The corrected numerical equation

The corrected operational matrix equation is

$$
\boxed{
\begin{aligned}
\mathbf c^\top\mathbf M(t)
={}&y_0\mathbf e^\top\mathbf M(t)
+\mathbf c_b^\top\mathbf P_\alpha(t)\mathbf M(t)\\
&+\mathbf c_\sigma^\top\mathbf S_\alpha\mathbf M(t)
-\operatorname{Trace}_n(t).
\end{aligned}
}
$$

The vector $\mathbf e$ represents the constant function $1$ in the chosen basis, so $y_0\mathbf e^\top\mathbf M(t)=y_0$.

At the collocation points, the stochastic-equation loss becomes

$$
L_{\mathrm{SDE}}
=\frac{1}{N_c}\sum_{i=1}^{N_c}
\left[
\begin{aligned}
&\mathbf c^\top\mathbf M(t_i)
-y_0\mathbf e^\top\mathbf M(t_i)\\
&-\mathbf c_b^\top\mathbf P_\alpha(t_i)\mathbf M(t_i)
-\mathbf c_\sigma^\top\mathbf S_\alpha\mathbf M(t_i)
+\operatorname{Trace}_n(t_i)
\end{aligned}
\right]^2.
$$

The trace is written as a drift-side correction. One could rearrange the same equation differently, but this placement has two practical advantages:

- $\mathbf S_\alpha$ remains a reusable, problem-independent stochastic matrix;
- the trace remains visibly tied to the particular diffusion function and fitted optimizer.

---

## 10. Important special cases and cautions

### 10.1 Additive noise

Noise is **additive** when

$$
\sigma(t,y)=\sigma_0(t),
$$

so it does not depend on the state $y$. If its basis projection is imposed exactly, then $\mathbf c_\sigma^*$ is deterministic. Therefore,

$$
D_s\mathbf c_\sigma^*=0
\qquad\Longrightarrow\qquad
\operatorname{Trace}_n(t)=0.
$$

This agrees with the familiar fact that Itô and Stratonovich integrals have no conversion correction for additive noise.

There is a numerical caution. If $\mathbf c_\sigma$ is fitted jointly with all other coefficients using finite penalty weights, the optimizer may allow the diffusion fit to depend spuriously on $\mathbf S_\alpha$. A nonzero trace in a truly additive model is then a discretization or optimization artifact. Additive noise is therefore a valuable regression test: either enforce its deterministic diffusion projection separately, or check that the computed trace is negligible.

Affine multiplicative noise,

$$
\sigma(t,y)=\sigma_0(t)+\sigma_1y,
\qquad \sigma_1\ne0,
$$

is not additive. Its fitted diffusion coefficients remain random because they depend on the fitted state.

### 10.2 Self-consistency

The trace depends on $D_s\mathbf c_\sigma^*$, which depends on the optimizer. But adding the trace to the residual changes the optimizer. Therefore, the corrected solution and its trace should ideally be determined together.

Two approaches are possible:

- **Frozen-trace or one-step correction:** solve the uncorrected problem, compute its exact finite-dimensional trace, and subtract that trace once. This is an exact diagnostic of the uncorrected optimizer but only a one-step correction of the numerical scheme.
- **Self-consistent correction:** repeatedly solve the corrected residual and update the sensitivity and trace, for example with an outer fixed-point iteration, until both stop changing.

These two procedures should not be described as mathematically identical.

---

## 11. What happens when $\alpha=1$?

When $\alpha=1$,

$$
K_1(t,s)=\mathbf 1_{\{s<t\}},
$$

and the SVIE becomes a standard stochastic differential equation.

If the spectral approximation converges in the symmetric Wong--Zakai sense, and if both the diffusion approximation and its Malliavin derivative converge appropriately, then

$$
\operatorname{Trace}_n(t)
\longrightarrow
\frac12\int_0^t
\sigma(s,y(s))\,\partial_y\sigma(s,y(s))\,ds.
$$

Here $\partial_y\sigma$ is the ordinary derivative of $\sigma(t,y)$ with respect to its state argument $y$. The corresponding local drift is

$$
\boxed{
b_{\mathrm{eff}}(t,y)
=b(t,y)-\frac12\sigma(t,y)\partial_y\sigma(t,y).
}
$$

This is the familiar Stratonovich-to-Itô drift correction, with the sign appropriate when converting an uncorrected symmetric approximation into the target Itô equation.

The convergence statement is conditional: the finite-dimensional identity by itself does not prove the required Wong--Zakai and Malliavin--Sobolev convergence.

### 11.1 Do not integrate the correction twice

The trace

$$
\frac12\int_0^t\sigma\,\partial_y\sigma\,ds
$$

is already accumulated over time. By contrast,

$$
-\frac12\sigma(t,y)\partial_y\sigma(t,y)
$$

is a local drift coefficient that will later be integrated by the equation. Adding an extra factor of $t$ to the local coefficient integrates the time dependence twice and gives the wrong correction.

For geometric Brownian motion,

$$
dy=\mu y\,dt+\sigma y\,dB_t,
$$

we have $\sigma(t,y)=\sigma y$ and

$$
\sigma(t,y)\partial_y\sigma(t,y)=\sigma^2y.
$$

Thus the correct local modification is

$$
b_{\mathrm{eff}}(t,y)
=\mu y-\frac12\sigma^2y,
$$

not $\mu y-(t/2)\sigma^2y$.

---

## 12. Why the fractional case is harder

For $\alpha<1$, the memory kernel is singular and the resulting process is generally outside the classical semimartingale setting. There is no automatic universal formula of the form

$$
b_{\mathrm{eff}}(t,y)
=b(t,y)-C_\alpha(t)\sigma(t,y)\partial_y\sigma(t,y).
$$

Such a scalar coefficient $C_\alpha(t)$ is justified only if one separately proves that

$$
\operatorname{Trace}_n(t)
=\int_0^tK_\alpha(t,s)C_\alpha(s)
\sigma(s,y_n(s))\partial_y\sigma(s,y_n(s))\,ds
$$

to the required accuracy.

Without that proof, the safe procedure is to evaluate the finite-dimensional trace directly:

$$
\operatorname{Trace}_n(t)
=\int_0^tK_\alpha(t,s)
(D_s\mathbf c_\sigma^*)^\top\mathbf M(s)\,ds.
$$

In other words, the familiar factor $1/2$ from ordinary SDEs should not simply be copied into the fractional problem.

---

## 13. When does the corrected term converge to the Itô integral?

Define

$$
u_n(t,s)
=K_\alpha(t,s)(\mathbf c_{\sigma,n}^*)^\top\mathbf M_n^\Lambda(s),
$$

and let

$$
u(t,s)=K_\alpha(t,s)\sigma(s,y(s))
$$

be the exact stochastic integrand.

For a fixed $t$, assume:

1. **Malliavin--Sobolev convergence:**

   $$
   u_n(t,\cdot)\to u(t,\cdot)
   \quad\text{in }\mathbb D^{1,2}(L^2(0,t)).
   $$

   This requires convergence of both the fitted integrand and its Malliavin derivative. It is stronger than ordinary mean-square convergence.

2. **Vanishing stochastic projection error:**

   $$
   (\mathbf c_{\sigma,n}^*)^\top\boldsymbol\varepsilon_n(t)
   \to0
   \quad\text{in }L^2(\Omega).
   $$

   Here $L^2(\Omega)$ means square integrability over the probability space: $\mathbb E[|X|^2]<\infty$.

The divergence operator is closed, meaning that convergence of suitable integrands and their divergences preserves the divergence relation. Under the two assumptions above,

$$
\mathcal A_n(t)-\operatorname{Trace}_n(t)
\longrightarrow
\delta(u(t,\cdot)).
$$

Because the exact integrand $u(t,s)$ is adapted, its divergence integral is the target Itô integral:

$$
\delta(u(t,\cdot))
=\frac1{\Gamma(\alpha)}
\int_0^t(t-s)^{\alpha-1}
\sigma(s,y(s))\,dB_s.
$$

This result is deliberately conditional. Mean-square convergence of $y_n$ alone does not control $D_su_n$, and completeness of the deterministic basis alone does not guarantee that the random projection remainder vanishes.

---

## 14. What should be checked numerically?

As the basis degree $n$ increases, monitor these three quantities separately:

$$
\|\operatorname{Trace}_n\|_{L^2(0,1)},
\qquad
\|\boldsymbol\varepsilon_n\|_{L^2(0,1)},
\qquad
\kappa(\mathbf K^\top\mathbf K).
$$

They measure different issues:

- $\|\operatorname{Trace}_n\|_{L^2(0,1)}$ measures the overall size of the Malliavin correction over time.
- $\|\boldsymbol\varepsilon_n\|_{L^2(0,1)}$ measures stochastic projection error.
- $\kappa(\mathbf K^\top\mathbf K)$ is the condition number of the normal-equation matrix. A very large value indicates that the coefficient sensitivity may be numerically unstable.

The trace does not necessarily approach zero as $n$ increases. A nonzero limiting trace can represent a permanent structural correction caused by the global, anticipative approximation rather than a temporary discretization error.

---

## 15. The full idea in plain language

1. The global solver fits the whole Brownian path at once.
2. Its fitted diffusion coefficients can therefore depend on future Brownian values.
3. This makes the fitted diffusion non-adapted.
4. The numerical matrix product is consequently not automatically the desired Itô integral.
5. Malliavin differentiation measures the fitted coefficients' sensitivity to every point of the Brownian path.
6. Differentiating the optimizer's normal equations gives that sensitivity exactly in the finite-dimensional model.
7. Integrating the sensitivity against the fractional kernel gives the Malliavin trace.
8. Subtracting the trace gives the correct divergence-integral term, apart from a separate stochastic projection remainder.
9. If the limiting integrand is adapted and the stronger convergence assumptions hold, the corrected term converges to the desired Itô integral.

The practical formula to remember is

$$
\boxed{
\operatorname{Trace}_n(t)
=\int_0^tK_\alpha(t,s)
(D_s\mathbf c_\sigma^*)^\top\mathbf M(s)\,ds,
\qquad
\mathcal A_n^{\mathrm{corrected}}(t)
=\mathcal A_n(t)-\operatorname{Trace}_n(t).
}
$$

---

## 16. Short glossary

**Almost everywhere (a.e.).** A statement holds almost everywhere if it may fail on a set of time points having total length zero. Such exceptional points do not affect the integrals used here.

**Basis.** A collection of functions used as building blocks for an approximation. Truncating after $m$ functions turns an infinite-dimensional function problem into a finite-dimensional coefficient problem.

**Brownian motion.** A continuous random process $B_t$ with $B_0=0$, independent increments, and $B_t-B_s\sim N(0,t-s)$ for $t>s$. The notation $N(0,t-s)$ means a normal random variable with mean $0$ and variance $t-s$.

**Collocation.** A numerical strategy that requires an approximate equation to hold at selected time points rather than at every time in the interval.

**Condition number.** A measure of how strongly small input or roundoff errors can be amplified when solving a linear system. A condition number near $1$ is favorable; a very large value signals possible instability.

**Convergence in $L^2(\Omega)$.** Random variables $X_n$ converge to $X$ in $L^2(\Omega)$ when $\mathbb E[|X_n-X|^2]\to0$. This is also called mean-square convergence. The symbol $\Omega$ represents the set of all possible random outcomes or Brownian paths.

**Convergence in $L^2(0,1)$.** Functions $f_n$ converge to $f$ in $L^2(0,1)$ when $\int_0^1|f_n(t)-f(t)|^2dt\to0$. Unlike $L^2(\Omega)$, this norm averages over time rather than random outcomes.

**Deterministic.** Non-random. A deterministic quantity has the same value for every Brownian path.

**Finite-dimensional.** Described by finitely many numbers. At degree $n$, this method represents each fitted function using $m=n+1$ coefficients.

**Full column rank.** The columns of a matrix are linearly independent. For $\mathbf K$, this ensures that the least-squares coefficient vector is locally unique and that $\mathbf K^\top\mathbf K$ is invertible.

**Global or spectral approximation.** An approximation using basis functions that extend across the whole time interval. “Spectral” refers to increasing the number of basis modes rather than refining only small local time steps.

**Gram matrix.** A matrix of pairwise basis inner products. If the basis functions are linearly independent, its Gram matrix is positive definite and invertible.

**Hessian.** The matrix of all second partial derivatives of a scalar objective. It describes the local curvature of the least-squares loss near an optimizer.

**Itô integral.** A stochastic integral defined using left-endpoint information. Its integrand must be adapted, so it cannot use future Brownian values.

**Least squares.** A fitting method that chooses coefficients to minimize the sum of squared residuals.

**Malliavin--Sobolev convergence.** Convergence of random quantities together with convergence of their Malliavin derivatives. It controls both values and sensitivity to the Brownian path.

**Normal equations.** The first-order equations $\mathbf J^\top\mathbf r=0$ for a nonlinear least-squares problem, or $\mathbf K^\top\mathbf K\boldsymbol\theta=\mathbf K^\top\mathbf d$ for a linear one.

**Operational matrix.** A matrix that approximates the action of an operator, such as integration, on basis coefficients. It replaces repeated function-level integration with matrix algebra.

**Optimizer map.** The rule that maps problem data, here including $\mathbf S_\alpha$, to the fitted minimizer $\boldsymbol\theta^*$.

**Outer product and rank one.** For column vectors $a$ and $b$, the matrix $ab^\top$ is their outer product. All its columns are multiples of $a$, so its rank is at most one.

**Probability space.** The mathematical model $(\Omega,\mathcal F,\mathbb P)$ of random outcomes: $\Omega$ is the outcome set, $\mathcal F$ is the collection of measurable events, and $\mathbb P$ assigns probabilities to those events.

**Projection.** The best approximation, under a chosen norm, of an object by elements of a finite basis space. The projection remainder is the part the finite space cannot represent.

**Regularization or penalty weight.** An added loss term, multiplied by a chosen weight, that encourages a constraint or stabilizes a fit without necessarily enforcing it exactly.

**Residual.** The difference between the left- and right-hand sides of an equation after substituting the numerical approximation. A zero residual means the equation is satisfied exactly at the tested point.

**Semimartingale.** A broad class of stochastic processes for which classical Itô calculus is defined. Standard SDE solutions are usually semimartingales; fractional-memory processes need not be.

**Skorokhod integral.** An extension of the Itô integral that can accept certain non-adapted integrands. It is defined as the adjoint of the Malliavin derivative.

**Stratonovich integral.** A stochastic integral associated with symmetric, midpoint-style approximations. For state-dependent noise it differs from the Itô integral by a drift correction.

**Wong--Zakai limit.** The limit obtained when Brownian motion is replaced by smooth path approximations and the corresponding ordinary differential equations are solved. Symmetric smooth approximations typically converge to the Stratonovich interpretation, which is why the classical half-correction appears when $\alpha=1$.
