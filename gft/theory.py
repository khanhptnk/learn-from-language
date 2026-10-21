"""Numerical checks of the results in GFT_revised_2026-10-20.pdf (General Feedback Training).

Everything here works on small enumerated instances: executions x in X, feedback f in F, a channel
F[x, f] = F(f | x), an ideal-feedback law g[f] = G*(f), and execution laws p over X. Section and
equation numbers refer to the revised paper.
"""

import numpy as np
from scipy.optimize import linprog
from scipy.special import expit, logit as _logit

from gft.algos.gft_exact import feedback_marginal, gft_step
from gft.metrics import kl, tv


def gft_operator(p, F, g):
    """Exact population GFT update T(p) (Lemma 5.1, Eq. 12)."""
    return gft_step(p, F, g)[0]


def posterior(p, F, f):
    """Bayes posterior r(x | f) = p(x) F(f | x) / m_p(f) (Eq. 11)."""
    w = p * F[:, f]
    return w / w.sum()


# ---------------------------------------------------------------- separation (Def. 6.1)

def separation_gamma(F, optimal):
    """Largest gamma for which some statistic s: F -> [0, 1] is gamma-optimality separating (Eq. 18).

    LP over (s, a, gamma): E[s | x] = a on C, E[s | x] <= a - gamma off C, 0 <= s, a <= 1.
    Returns (gamma, s); gamma <= 0 means no separating statistic exists.
    """
    n_x, n_f = F.shape
    c = np.zeros(n_f + 2)
    c[-1] = -1.0                                         # maximize gamma
    A_eq = np.hstack([F[optimal], -np.ones((optimal.sum(), 1)), np.zeros((optimal.sum(), 1))])
    A_ub = np.hstack([F[~optimal], -np.ones(((~optimal).sum(), 1)), np.ones(((~optimal).sum(), 1))])
    res = linprog(c, A_ub=A_ub if len(A_ub) else None, b_ub=np.zeros(len(A_ub)) if len(A_ub) else None,
                  A_eq=A_eq, b_eq=np.zeros(optimal.sum()),
                  bounds=[(0, 1)] * n_f + [(0, 1), (None, 1)], method="highs")
    return float(-res.fun), res.x[:n_f]


# ---------------------------------------------------------------- damped operator (Prop. C.1)

def damped_run(p0, F, g, eta, T):
    ps = [np.asarray(p0, float)]
    for _ in range(T):
        ps.append((1 - eta) * ps[-1] + eta * gft_operator(ps[-1], F, g))
    return np.stack(ps)


# ---------------------------------------------------------------- FCP (Section 10)

def fcp_population(rho0, F, G, T):
    """Exact-population online FCP: fit q_{t+1}(x | f) on (x ~ rho_t, f ~ F(. | x)), then roll out
    under desired feedback G: rho_{t+1}(x) = sum_f G(f) q_{t+1}(x | f) (Eqs. 48-50). Implemented through
    the conditional table to check, rather than assume, the equivalence with the GFT operator."""
    rhos = [np.asarray(rho0, float)]
    for _ in range(T):
        rho = rhos[-1]
        joint = rho[:, None] * F                         # (x, f)
        m = joint.sum(0)
        q = np.where(m > 0, joint / np.where(m > 0, m, 1), 0.0)
        rhos.append(q @ G)
    return np.stack(rhos)


def fcp_fixed_message_closed_form(rho0, F, f_plus, t):
    """Eq. 51: rho_t(x) proportional to rho_0(x) F(f+ | x)^t."""
    w = rho0 * F[:, f_plus] ** t
    return w / w.sum()


def conditioning_variance_identity(p, F, f_plus):
    """Eq. 47: m_{r_p(.|f+)}(f+) = m_p(f+) + Var_{X~p}[F(f+|X)] / m_p(f+). Returns (lhs, rhs)."""
    m = feedback_marginal(p, F)[f_plus]
    r = posterior(p, F, f_plus)
    lhs = r @ F[:, f_plus]
    var = p @ F[:, f_plus] ** 2 - m ** 2
    return float(lhs), float(m + var / m)


# ---------------------------------------------------------------- SDPO counterexample (Prop. 11.1)

C_SDPO = np.log(9.0)


def sigmoid(z):
    return expit(z)


def logit(u):
    return _logit(u)


def sdpo_two_action_step(u):
    """One exact population reverse-KL self-distillation step (Eq. 53)."""
    m_plus = 0.1 + 0.8 * u
    t = logit(u)
    return sigmoid(m_plus * (t + C_SDPO) + (1 - m_plus) * (t - C_SDPO))


def sdpo_reverse_kl_minimizer(u, grid=200001):
    """Brute-force minimizer of the one-step SDPO surrogate (Eq. 52) to check the closed form (53)."""
    m_plus = 0.1 + 0.8 * u
    qp, qm = sigmoid(logit(u) + C_SDPO), sigmoid(logit(u) - C_SDPO)
    v = np.linspace(1e-6, 1 - 1e-6, grid)
    bkl = lambda a, b: a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))
    loss = m_plus * bkl(v, qp) + (1 - m_plus) * bkl(v, qm)
    return float(v[np.argmin(loss)])


def two_action_channel():
    """Actions (g, b); feedback (+, -); F(+|g) = F(-|b) = 0.9. G* = F(.|g)."""
    F = np.array([[0.9, 0.1], [0.1, 0.9]])
    return F, F[0].copy(), np.array([True, False])


# ---------------------------------------------------------------- gradients (Section 9)

def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def J_value(theta, F, g):
    """J(theta) = -KL(G* || m_theta) for a softmax execution law p_theta = softmax(theta) (Eq. 43)."""
    return -kl(g, feedback_marginal(softmax(theta), F))


def gradient_forms(theta, F, g):
    """The three exact expressions of Prop. 9.1 plus the distillation surrogate gradient at theta_t.

    Score s_theta(x) = e_x - p_theta for a softmax over executions.
      (44)  E_{f~G*, x~r(.|f)} s(x)
      (45)  E_{x~p, f~F(.|x)} [G*(f)/m(f)] s(x)
      L_t   grad of E_{x~p+} log p_theta(x) at theta_t, with p+ = sum_f G*(f) r(. | f) held fixed
    """
    p = softmax(theta)
    n = len(p)
    S = np.eye(n) - p[None, :]                           # S[x] = score of execution x
    m = feedback_marginal(p, F)
    on = g > 0
    g44 = np.zeros(n)
    for f in np.nonzero(on)[0]:
        g44 += g[f] * posterior(p, F, f) @ S
    W = F[:, on] @ (g[on] / m[on])                       # W(x) = sum_f F(f|x) G*(f)/m(f)
    g45 = (p * W) @ S
    p_plus = gft_operator(p, F, g)
    g_sur = p_plus - p                                   # gradient of sum_x p+(x) log softmax(theta)_x
    return g44, g45, g_sur


def finite_difference(fun, theta, eps=1e-6):
    out = np.zeros_like(theta)
    for i in range(len(theta)):
        d = np.zeros_like(theta)
        d[i] = eps
        out[i] = (fun(theta + d) - fun(theta - d)) / (2 * eps)
    return out


def approximate_posterior_bias(theta, F, g, r_phi):
    """Prop. 9.2: returns (||g(theta, phi) - grad J||, 2 B delta) for learned posteriors r_phi[f] (rows)."""
    p = softmax(theta)
    S = np.eye(len(p)) - p[None, :]
    B = np.linalg.norm(S, axis=1).max()
    on = np.nonzero(g > 0)[0]
    g_phi = sum(g[f] * r_phi[f] @ S for f in on)
    g_true = gradient_forms(theta, F, g)[0]
    delta = sum(g[f] * tv(r_phi[f], posterior(p, F, f)) for f in on)
    return float(np.linalg.norm(g_phi - g_true)), float(2 * B * delta)


# ---------------------------------------------------------------- rejection sampling (Appendix F)

def rejection_sample(rng, p, F, f_star, n):
    """Draw X ~ p, F ~ F(.|X) until F == f*; returns accepted executions and the number of trials each."""
    xs, trials = [], []
    for _ in range(n):
        k = 0
        while True:
            k += 1
            x = rng.choice(len(p), p=p)
            if rng.choice(F.shape[1], p=F[x]) == f_star:
                break
        xs.append(x)
        trials.append(k)
    return np.array(xs), np.array(trials)


# ---------------------------------------------------------------- finite-tabular Batch GFT (Thm 8.1)

def batch_gft_tabular(rng, p0, F, g, T, N, M):
    """Algorithm 2 with unrestricted tables and exact empirical fits on FRESH batches.

    N, M: functions t -> batch sizes. Stage I fits the empirical conditional r_hat(x | f) from N_t
    (x, f) pairs of the current law; for ideal-supported f never observed, r_hat(. | f) is the
    current law p_t (any choice is allowed by the theorem). Stage II draws M_t executions from
    sum_f G*(f) r_hat(. | f) and sets p_{t+1} to their empirical law.
    """
    n_x, n_f = F.shape
    ps = [np.asarray(p0, float)]
    for t in range(T):
        p = ps[-1]
        x = rng.choice(n_x, size=N(t), p=p)
        u = rng.random(N(t))
        f = (F[x].cumsum(1) < u[:, None]).sum(1).clip(max=n_f - 1)
        joint = np.zeros((n_x, n_f))
        np.add.at(joint, (x, f), 1)
        col = joint.sum(0)
        r_hat = np.where(col > 0, joint / np.where(col > 0, col, 1), p[:, None])
        q_hat = r_hat @ g
        xs = rng.choice(n_x, size=M(t), p=q_hat / q_hat.sum())
        ps.append(np.bincount(xs, minlength=n_x) / M(t))
    return np.stack(ps)


def thm81_constants(pbar, p0, n_f):
    """Constants of Theorem 8.1 (Eq. 31) for a fixed request: b, beta and the bound (v + 7/4)/(T+1)."""
    S = pbar > 0
    v = kl(pbar, p0)
    h = float(-(pbar[S] * np.log(pbar[S])).sum())
    a = float(pbar[S].min())
    b = np.exp(-(v + 1 + h) / a)
    beta = b * a / (2 * n_f)
    return dict(v=v, h=h, a=a, b=b, beta=beta)


def thm81_batch_sizes(c, n_x, n_f, t, delta):
    """N_t, M_t of Eqs. 32-33 (usually astronomically large; reported, not run)."""
    e_t = c["beta"] / (4 * (t + 1) ** 3)
    log_term = np.log(2 * (t + 1) * (t + 2) / delta)
    N = 8 / (c["b"] ** 2 * e_t ** 2) * ((n_x * n_f + 1) * np.log(2) + log_term)
    M = 2 / e_t ** 2 * ((n_x + 1) * np.log(2) + log_term)
    return N, M


# ---------------------------------------------------------------- linear-softmax Batch GFT (Thm 8.2)

def linear_softmax_batch_gft(rng, theta0, T, N, M, bound=6.0):
    """Section 8.2 example: one step, actions (1, 0) with p_theta(1) = sigmoid(theta), F(+|1) = 0.9,
    F(+|0) = 0.1, G* = F(.|1). Auxiliary logistic model logit r_phi(1 | f) = phi0 + phi1 * (+1 if f = + else -1),
    which realizes the exact posterior (phi0 = theta, phi1 = log 9). Both fits are exact empirical MLEs on
    fresh batches, projected onto [-bound, bound] (the compact parameter sets of the theorem).
    Returns theta_t, phi_t and the projected-gradient residual |theta - Pi(theta + grad J)|.
    """
    thetas, phis, resid = [theta0], [], []
    for t in range(T):
        th = thetas[-1]
        x = rng.random(N(t)) < sigmoid(th)                         # action 1 ?
        plus = rng.random(N(t)) < np.where(x, 0.9, 0.1)
        # Stage I: logistic regression of x on the sign feature, solved exactly (two cells).
        logits = []
        for s in (True, False):
            k = plus == s
            freq = (x[k].sum() + 0.5) / (k.sum() + 1.0)            # tiny smoothing keeps logits finite
            logits.append(logit(freq))
        phi = np.clip([(logits[0] + logits[1]) / 2, (logits[0] - logits[1]) / 2], -bound, bound)
        phis.append(phi)
        # Stage II: f* ~ G* = (0.9, 0.1); x+ ~ r_phi(. | f*); student MLE = logit of the empirical frequency.
        fstar = rng.random(M(t)) < 0.9
        xp = rng.random(M(t)) < sigmoid(phi[0] + phi[1] * np.where(fstar, 1, -1))
        freq = (xp.sum() + 0.5) / (M(t) + 1.0)
        thetas.append(float(np.clip(logit(freq), -bound, bound)))
        resid.append(abs(th - np.clip(th + dJ_two_action(th), -bound, bound)))
    return np.array(thetas), np.array(phis), np.array(resid)


def dJ_two_action(theta):
    """d/dtheta of J(theta) = -KL(G* || m_theta) for the Section 8.2 example."""
    u = sigmoid(theta)
    m_plus = 0.1 + 0.8 * u
    dm = 0.8 * u * (1 - u)
    return 0.9 * dm / m_plus - 0.1 * dm / (1 - m_plus)


# ---------------------------------------------------------------- Online GFT (Thm D.1)

def online_gft(rng, F, g, optimal, T, beta0=0.5, alpha0=0.05, bound=8.0, theta0=None, log_every=1000):
    """Algorithm 3 with single samples, tabular softmax actor theta[x] and auxiliary logits psi[f, x]
    (r_phi(x | f) = softmax(psi[f])), projected onto [-bound, bound]; steps beta_t = beta0 (t+1)^-0.6,
    alpha_t = alpha0 (t+1)^-0.9 (Eq. 78). Logs the tracking error E_{f~G*} TV(r_phi(.|f), r_theta(.|f))
    and |grad J(theta)| every `log_every` iterations, as rows (t, tracking, |grad J|, p_theta(C))."""
    n_x, n_f = F.shape
    theta = np.zeros(n_x) if theta0 is None else np.array(theta0, float)
    psi = np.zeros((n_f, n_x))
    log = []
    for t in range(T):
        beta, alpha = beta0 * (t + 1) ** -0.6, alpha0 * (t + 1) ** -0.9
        p = softmax(theta)
        x = rng.choice(n_x, p=p)
        f = rng.choice(n_f, p=F[x])
        r = softmax(psi[f])
        grad = -r
        grad[x] += 1.0                                             # d log r(x | f) / d psi[f]
        psi[f] = np.clip(psi[f] + beta * grad, -bound, bound)
        fs = rng.choice(n_f, p=g)
        xp = rng.choice(n_x, p=softmax(psi[fs]))
        s = -p
        s[xp] += 1.0
        theta = theta + alpha * s
        if t % log_every == 0 or t == T - 1:
            track = sum(g[k] * tv(softmax(psi[k]), posterior(p, F, k)) for k in np.nonzero(g > 0)[0])
            gJ = np.linalg.norm(gradient_forms(theta, F, g)[0])
            log.append((t, track, gJ, float(p[optimal].sum())))
    return theta, psi, np.array(log)


# ---------------------------------------------------------------- restricted-actor stationary point (Section D.3)

def d3_counterexample(theta=0.0):
    """p(success) = sigmoid(theta^2), deterministic binary feedback, G* = "success": J = log sigmoid(theta^2).
    Returns (dJ/dtheta, success probability) at theta."""
    return float(2 * theta * (1 - sigmoid(theta ** 2))), float(sigmoid(theta ** 2))
