## Thesis context

Bachelor thesis in mathematical finance on inference for the population Sharpe ratio under non-i.i.d. returns, especially serial dependence and conditional heteroskedasticity. Main aim: replace naive i.i.d.-normal inference with asymptotic theory that still works under stationary dependent returns, and derive closed-form long-run variance for symmetric GARCH(1,1).

---

## Core setup

Observed excess returns: ((X_t)_{t\ge1}), strictly stationary and ergodic.

Moments:
[
\mu=\E[X_t],\qquad m_2=\E[X_t^2],\qquad \sigma^2=m_2-\mu^2>0,\qquad S=\mu/\sigma.
]

Sample estimators:
[
\hat\mu_T=T^{-1}\sum_{t=1}^T X_t,\qquad
\hat m_{2,T}=T^{-1}\sum_{t=1}^T X_t^2,\qquad
\hat\sigma_T^2=\hat m_{2,T}-\hat\mu_T^2,\qquad
\hat S_T=\hat\mu_T/\hat\sigma_T.
]

Interpretation: (S) is the population Sharpe ratio of excess returns.

---

## Influence-function / von Mises layer

View Sharpe as a functional of the marginal law (P):
[
\mu(P)=\int x,dP(x),\qquad
m_2(P)=\int x^2,dP(x),\qquad
\sigma^2(P)=m_2(P)-\mu(P)^2,\qquad
S(P)=\mu(P)/\sigma(P).
]

Contaminated law at point (z):
[
P_\varepsilon=(1-\varepsilon)P+\varepsilon\delta_z.
]

Influence function:
[
\operatorname{IF}(z;F,P)=\left.\frac{d}{d\varepsilon}F(P_\varepsilon)\right|_{\varepsilon=0}.
]

Known examples:
[
\operatorname{IF}(z;\mu,P)=z-\mu(P),
]
[
\operatorname{IF}(z;S,P)
========================

\frac{z-\mu}{\sigma}
-\frac{S}{2}\left(\frac{(z-\mu)^2}{\sigma^2}-1\right).
]

Define standardized return
[
y_t=\frac{X_t-\mu}{\sigma},
]
then the Sharpe influence function evaluated at (X_t) is
[
\psi_t
======

y_t-\frac{S}{2}(y_t^2-1).
]

---

## First-order expansion of the Sharpe estimator

Let
[
g(u,v)=u/\sqrt{v-u^2},\qquad \eta=(\mu,m_2),\qquad \hat\eta_T=(\hat\mu_T,\hat m_{2,T}),
]
so (\hat S_T=g(\hat\eta_T)), (S=g(\eta)).

Taylor/von Mises expansion:
[
\hat S_T-S
==========

\frac1T\sum_{t=1}^T\psi_t+r_T,
\qquad
r_T=o_p(T^{-1/2}),
]
hence
[
\sqrt{T}(\hat S_T-S)
====================

\frac1{\sqrt T}\sum_{t=1}^T\psi_t+o_p(1).
]

Equivalent linear representation in ((X_t,X_t^2)):
[
\psi_t=c^\top Z_t,
\qquad
Z_t=
\begin{pmatrix}
X_t-\mu\
X_t^2-m_2
\end{pmatrix},
\qquad
c=
\begin{pmatrix}
m_2/\sigma^3\
-\mu/(2\sigma^3)
\end{pmatrix}.
]

---

## Main asymptotic theorem under dependence

Assume:

1. ((X_t)) is strictly stationary and ergodic.
2. (\E|X_t|^{4+\delta}<\infty) for some (\delta>0).
3. High-level CLT:
   [
   \frac1{\sqrt T}\sum_{t=1}^T
   \begin{pmatrix}
   X_t-\mu\
   X_t^2-m_2
   \end{pmatrix}
   \Rightarrow
   \mathcal N(0,\Sigma).
   ]

Then
[
\sqrt T(\hat S_T-S)\Rightarrow \mathcal N(0,\Omega_\psi),
\qquad
\Omega_\psi=c^\top\Sigma c.
]

If (\Sigma=\sum_{k=-\infty}^{\infty}\Gamma_Z(k)) with absolute convergence, then
[
\Omega_\psi=\sum_{k=-\infty}^{\infty}\gamma_\psi(k),
\qquad
\gamma_\psi(k)=\C(\psi_t,\psi_{t-k}),
]
so (\Omega_\psi) is the long-run variance of the influence-function process.

Important point: stationarity + ergodicity do **not** imply the CLT by themselves; the CLT is assumed at high level.

---

## IID special case

If (X_t) are i.i.d., (\E|X_1|^4<\infty), and
[
y_1=\frac{X_1-\mu}{\sigma},\qquad
\gamma_3=\E[y_1^3],\qquad
\gamma_4=\E[y_1^4],
]
then
[
\sqrt T(\hat S_T-S)\Rightarrow \mathcal N(0,\Omega),
\qquad
\Omega
======

1-\gamma_3S+\frac{\gamma_4-1}{4}S^2.
]

Gaussian i.i.d. case: (\gamma_3=0,\gamma_4=3), so
[
\Omega=1+\frac12S^2.
]

This yields the analytic i.i.d.-normal Wald CI:
[
\hat S_T \pm z_{1-\alpha/2}\sqrt{(1+\hat S_T^2/2)/T}.
]

---

## Symmetric GARCH(1,1) model

Model:
[
X_t=\mu+\sigma_t\varepsilon_t,
\qquad
\sigma_t^2=\alpha_0+\alpha_1(X_{t-1}-\mu)^2+\beta\sigma_{t-1}^2,
]
with
[
\alpha_0>0,\quad \alpha_1\ge0,\quad \beta\ge0.
]

Innovations ((\varepsilon_t)) are i.i.d., symmetric, independent of (\mathcal F_{t-1}), and satisfy
[
\E[\varepsilon_t]=0,\qquad \E[\varepsilon_t^2]=1,\qquad h_2=\E[\varepsilon_t^4]<\infty.
]

Let
[
\gamma=\alpha_1+\beta,
\qquad
d=1-\gamma^2-(h_2-1)\alpha_1^2.
]

Assume (d>0). This ensures finite fourth moments and in particular (\gamma<1), hence finite variance.

In this model, define
[
y_t=(X_t-\mu)/\sigma,\qquad
\psi_t=y_t-\frac S2(y_t^2-1),
\qquad
\Omega_G:=\sum_{k=-\infty}^{\infty}\C(\psi_t,\psi_{t-k}).
]

Then
[
\sqrt T(\hat S_T-S)\Rightarrow \mathcal N(0,\Omega_G),
]
with closed form
[
\Omega_G
========

1+\frac{S^2}{4}\cdot
\frac{(h_2-1)(1+\gamma)(1-\beta)^2}{d(1-\gamma)}.
]

Key proof structure:

* (y_t) is a martingale difference.
* By symmetry, mixed odd-square covariance terms vanish.
* Only square-square dependence remains.
* GARCH gives geometric decay in (\C(y_t^2,y_{t-k}^2)).
* Summing gives the closed form above.

So in this thesis, (\Omega_G) is the model-specific long-run variance for the Sharpe estimator in symmetric GARCH(1,1).

---

## Basic conditional scale / GARCH theory used later

Conditional scale model:
[
a_t=\sigma_t\varepsilon_t,\qquad X_t=\mu+a_t,
]
with (\sigma_t>0) and (\sigma_t) (\mathcal F_{t-1})-measurable, (\varepsilon_t) i.i.d. with mean (0), variance (1).

Then
[
\E[X_t\mid\mathcal F_{t-1}]=\mu,\qquad
\V(X_t\mid\mathcal F_{t-1})=\sigma_t^2.
]

GARCH(1,1) with constant mean:
[
X_t=\mu+\sigma_t\varepsilon_t,
\qquad
\sigma_t^2=\alpha_0+\alpha_1 a_{t-1}^2+\beta_1\sigma_{t-1}^2.
]

If second-moment stationarity holds ((\alpha_1+\beta_1<1)), then
[
\E[\sigma_t^2]=\frac{\alpha_0}{1-\alpha_1-\beta_1},
\qquad
\V(X_t)=\E[\sigma_t^2].
]

Standardized Student-(t_\nu) innovations:
if (Z\sim t_\nu), (\nu>2), define
[
\varepsilon=Z\sqrt{\frac{\nu-2}{\nu}},
]
so (\E[\varepsilon]=0), (\E[\varepsilon^2]=1). If (\nu>4), then (\E[\varepsilon^4]<\infty).

For standardized Student-(t_\nu),
[
h_2(\nu)=\E[\varepsilon_t^4]=3\frac{\nu-2}{\nu-4},\qquad \nu>4.
]
For Gaussian innovations, (h_2=3).

---

## Inference theory

### General asymptotic Wald CI

If
[
\sqrt T(\hat S_T-S)\Rightarrow\mathcal N(0,\Omega_\psi)
]
and there is a consistent estimator
[
\hat\Omega_\psi\to_p\Omega_\psi>0,
]
then
[
C_T(X)=
\left[
\hat S_T-z_{1-\alpha/2}\sqrt{\hat\Omega_\psi/T},
\ \hat S_T+z_{1-\alpha/2}\sqrt{\hat\Omega_\psi/T}
\right]
]
is an asymptotic ((1-\alpha))-CI:
[
\Pr(S\in C_T(X))\to 1-\alpha.
]

This is the standard studentized Slutsky/Wald argument.

### GARCH plug-in CI

In the symmetric GARCH(1,1) case, let
[
\theta=(\mu,\alpha_0,\alpha_1,\beta,\eta),
]
where (\eta) indexes the innovation family and (h_2(\eta)=\E_\eta[\varepsilon_t^4]).

If (\hat\theta) is a consistent MLE, define
[
\hat\gamma=\hat\alpha_1+\hat\beta,\qquad
\hat h_2=h_2(\hat\eta),\qquad
\hat d=1-\hat\gamma^2-(\hat h_2-1)\hat\alpha_1^2,
]
and
[
\hat\Omega_G
============

1+\frac{\hat S_T^2}{4}\cdot
\frac{(\hat h_2-1)(1+\hat\gamma)(1-\hat\beta)^2}{\hat d(1-\hat\gamma)}.
]

Then (\hat\Omega_G\to_p\Omega_G) by continuity on ({d>0}), so the Wald CI
[
\hat S_T \pm z_{1-\alpha/2}\sqrt{\hat\Omega_G/T}
]
has asymptotic coverage (1-\alpha).

So the thesis currently has two analytic CI routes:

1. i.i.d. Gaussian plug-in,
2. symmetric GARCH(1,1) plug-in via estimated parameters and innovation fourth moment.

---

## Maximum likelihood layer

Used for fitting parametric heteroskedastic models.

Assume
[
X_t=\mu+\sigma_t(\theta)\varepsilon_t,
]
where (\sigma_t(\theta)>0) is (\mathcal F_{t-1})-measurable and (\varepsilon_t) are i.i.d. with density (f_\varepsilon), CDF (F_\varepsilon), independent of (\mathcal F_{t-1}).

Conditional law:
[
\Pr(X_t\le x\mid\mathcal F_{t-1})
=================================

F_\varepsilon!\left(\frac{x-\mu}{\sigma_t(\theta)}\right),
]
[
f_{X_t\mid\mathcal F_{t-1}}(x)
==============================

\frac1{\sigma_t(\theta)}
f_\varepsilon!\left(\frac{x-\mu}{\sigma_t(\theta)}\right).
]

Conditional likelihood:
[
L_T(\theta)=\prod_{t=1}^T f_{X_t\mid\mathcal F_{t-1}}(x_t).
]

Log-likelihood:
[
\ell_T(\theta)
==============

\sum_{t=1}^T
\log f_\varepsilon!\left(\frac{x_t-\mu}{\sigma_t(\theta)}\right)
----------------------------------------------------------------

\sum_{t=1}^T\log \sigma_t(\theta).
]

This MLE is used to estimate the GARCH parameters and innovation parameter entering (\hat\Omega_G).

---

## Monte Carlo theory already in thesis

Goal: evaluate coverage / rejection properties of Sharpe-ratio CIs.

For a fixed DGP and method, define event (E), e.g.
[
E={S_{\text{true}}\in \text{CI}}.
]
Indicator per replication:
[
I_r=\mathbf 1{E\text{ occurs in replication }r},\qquad r=1,\dots,R.
]

Then
[
I_1,\dots,I_R \stackrel{iid}{\sim}\mathrm{Bernoulli}(p),
\qquad
\hat p=\frac1R\sum_{r=1}^R I_r.
]

So
[
\E[\hat p]=p,\qquad
\V(\hat p)=\frac{p(1-p)}R,
]
and by CLT,
[
\sqrt R(\hat p-p)\Rightarrow\mathcal N(0,p(1-p)).
]

Approximate 95% Monte Carlo error:
[
|\hat p-p|\lesssim 1.96\sqrt{p(1-p)/R}.
]

Worst case is (p(1-p)\le 1/4), so to guarantee Monte Carlo error at most (\varepsilon),
[
R\ge \frac{1.96^2}{4\varepsilon^2}.
]

With (\varepsilon=0.004), this gives
[
R=60025.
]

So the thesis justifies (R=60{,}025) replications as a worst-case Monte Carlo precision choice.

---

## Simulation design already implied

DGPs used / intended:

* i.i.d. normal,
* GARCH(1,1) with standardized Student-(t) innovations.

Because Sharpe is scale-invariant, one can set
[
\sigma=1,\qquad \mu=S_{\text{true}}
]
in simple designs, though GARCH experiments may additionally vary scale.

Main dimensions varied:

* sample size (n),
* true Sharpe (S_{\text{true}}),
* dependence / DGP,
* possibly volatility scale.

---

## What the thesis mathematically delivers so far

1. Defines Sharpe ratio as a functional of the return law.
2. Derives its influence function.
3. Derives first-order von Mises expansion of (\hat S_T).
4. Proves asymptotic normality of (\hat S_T) under a high-level CLT for ((X_t,X_t^2)).
5. Gives the i.i.d. closed-form asymptotic variance.
6. Gives a closed-form long-run variance for symmetric GARCH(1,1).
7. Builds asymptotic Wald confidence intervals from consistent variance estimators.
8. Gives a plug-in GARCH variance estimator using MLE.
9. Sets up conditional likelihood for the parametric GARCH fit.
10. Derives Monte Carlo replication count from binomial/CLT error control.