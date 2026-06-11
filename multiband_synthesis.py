"""
Certified multiband filter synthesis after

    V. Lunot, F. Seyfert, S. Bila, A. Nasser,
    "Certified computation of optimal multiband filtering functions,"
    IEEE Trans. MTT, 56(1), pp. 105-112, 2008.

The real Zolotarev problem:

    given passband intervals I = U I_k  and stopband intervals J = U J_k,
    find real-coefficient polynomials F, P with deg F = nF, deg P <= nP
    maximising   M = min_{w in J}  |F(w)/P(w)| / w_J(w)
    subject to   |F(w)/P(w)|  <=  psi_I(w)  on I,

    where the stopband weight is  w_J = psi_J  where psi_J > 0  and 1
    where psi_J = 0.  This is the paper's Section V.D specification
    mechanism: the rejection target enters MULTIPLICATIVELY ("replace
    bound 1 [over E(l)] by psi"), so the optimum exceeds every band's
    target by the same dB margin (20 log10 M) and every band stays
    binding -- unlike an additive reading |D| >= M + psi_J, under which
    any band with psi_J < -M becomes vacuous and the "optimal" filter
    may drive its rejection to 0 dB.  With psi_J = 0 everywhere the
    weight is 1 and M = min |F/P|: the classical unweighted problem.

Strategy used here:

  * enumerate the finitely many sign combinations (sigma_I, sigma_J);
  * internally rescale omega into [-1,1] so that monomial evaluation is
    well conditioned even for wide stopbands;
  * for every sign combination, BISECT on M (robust) using the LP
    feasibility test of Section V of the paper;
  * at every LP solve, perform a Remez-style exchange: whenever the
    continuous constraints are violated between samples, add the
    worst offenders and resolve, until the solution is valid
    everywhere on I U J;
  * keep the sign combination with the largest certified M.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from itertools import product
from numpy.polynomial import chebyshev as npch


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def _chebyshev_nodes(a: float, b: float, n: int) -> np.ndarray:
    """n Chebyshev-Lobatto nodes in [a, b], INCLUDING the endpoints.

    First-kind nodes (cos((2k+1)pi/(2n))) leave a gap near both endpoints.
    For stopband sampling that gap lets the LP park a reflection zero of
    F *exactly on the truncation boundary* where the sign constraint is
    not tested -- yielding spurious "optimal" filters with zeros on the
    interval edge.  Lobatto nodes close that loophole.
    """
    if n < 2:
        return np.array([0.5 * (a + b)])
    k = np.arange(n)
    x = np.cos(np.pi * k / (n - 1))
    return 0.5 * (a + b) + 0.5 * (b - a) * x


def _poly_eval(coeffs, x):
    """Evaluate ascending-order coefficients [c_0, ..., c_d] at x."""
    return np.polyval(np.asarray(coeffs)[::-1], x)


def _as_fn(v):
    if callable(v):
        return v
    return lambda _w, _v=v: _v


def _stopband_weight(psi):
    """Multiplicative stopband weight of the paper's Section V.D spec
    mechanism: the rejection target psi_J scales the criterion, with a
    weight of 1 where no target is given (psi_J = 0) so that the
    classical unweighted Zolotarev problem is recovered there."""
    return psi if psi > 1e-9 else 1.0


# -----------------------------------------------------------------------------
# LP assembly and feasibility test
# -----------------------------------------------------------------------------

def _cheb_basis_row(x, n):
    """Return [T_0(x), T_1(x), ..., T_{n-1}(x)] evaluated at x."""
    out = np.empty(n)
    if n >= 1:
        out[0] = 1.0
    if n >= 2:
        out[1] = x
    for k in range(2, n):
        out[k] = 2.0 * x * out[k - 1] - out[k - 2]
    return out


def _build_lp(I_pts, J_pts, nF, nP, M, psi_I_fn, psi_J_fn, coef_bound,
              F_ref=None, anchor=None):
    """Assemble the LP of Section V (paper) in the CHEBYSHEV basis.

       max h
        s.t.   sigma F(y) - M w_J(y) |P(y)| >= h |F_ref(y)|   for y in J
               sigma P(x) >= 0                                for x in I
               |F(x)| <= sigma psi(x) P(x)                    for x in I

    where w_J = psi_J where positive, else 1 (see _stopband_weight).

    Variables (Chebyshev on the rescaled interval):

        [f_0, ..., f_{nF},   p_0, ..., p_{nP-1},   h]

    with  F(t) = sum f_i T_i(t)   and   P(t) = T_nP(t) + sum_{i<nP} p_i T_i(t).

    The coefficient of T_nP in P is fixed at 1 (the classical
    Zolotarev lc(P)=1 normalisation, but in Chebyshev basis).
    Compactness is enforced by wide box bounds on the free coefficients.
    `F_ref` is the reference polynomial for the quadratic-correction
    denominator (eq. 14 of the paper); if None, we use rhs=1 (non-quadratic
    form, eq. 10).
    """
    n_f = nF + 1
    n_p = nP
    nvar = n_f + n_p + 1
    h_idx = nvar - 1

    c = np.zeros(nvar)
    c[h_idx] = -1.0

    A = []
    b = []

    for y, sigma in J_pts:
        # Multiplicative spec weighting (paper Section V.D): the level-M
        # constraint is sigma F - M w_J |P| >= h rhs.
        # The two-row linearisation below encodes sigma F - Meff |P| >= h
        # ONLY for Meff >= 0; with Meff < 0 the intersection of the two
        # rows would demand sigma F >= h + |Meff||P| -- a stricter, wrong
        # constraint that silently corrupts every probe at negative
        # levels.  The true constraint is vacuous there, so clamp: the
        # rows then just pin the assumed sign, sigma F >= h rhs.
        Meff = max(_stopband_weight(psi_J_fn(y)) * M, 0.0)
        Fvec = _cheb_basis_row(y, n_f)
        Pvec_all = _cheb_basis_row(y, nP + 1)
        Pvec_low = Pvec_all[:nP] if nP else np.empty(0)
        P_lead = Pvec_all[nP]

        # rhs for h (quadratic correction eq. 14)
        if F_ref is not None:
            rhs = max(abs(_cheb_eval(F_ref, y)), 1e-12)
        else:
            rhs = 1.0

        # sigma F - Meff P >= h * rhs
        row = np.zeros(nvar)
        row[:n_f] = -sigma * Fvec
        if n_p:
            row[n_f:n_f + n_p] = Meff * Pvec_low
        row[h_idx] = rhs
        A.append(row); b.append(-Meff * P_lead)

        # sigma F + Meff P >= h * rhs
        row = np.zeros(nvar)
        row[:n_f] = -sigma * Fvec
        if n_p:
            row[n_f:n_f + n_p] = -Meff * Pvec_low
        row[h_idx] = rhs
        A.append(row); b.append(Meff * P_lead)

    for x, sigma in I_pts:
        psi = psi_I_fn(x)
        Fvec = _cheb_basis_row(x, n_f)
        Pvec_all = _cheb_basis_row(x, nP + 1)
        Pvec_low = Pvec_all[:nP] if nP else np.empty(0)
        P_lead = Pvec_all[nP]

        # sigma P >= 0
        row = np.zeros(nvar)
        if n_p:
            row[n_f:n_f + n_p] = -sigma * Pvec_low
        A.append(row); b.append(sigma * P_lead)

        # F - sigma psi P <= 0
        row = np.zeros(nvar)
        row[:n_f] = Fvec
        if n_p:
            row[n_f:n_f + n_p] = -sigma * psi * Pvec_low
        A.append(row); b.append(sigma * psi * P_lead)

        # -F - sigma psi P <= 0
        row = np.zeros(nvar)
        row[:n_f] = -Fvec
        if n_p:
            row[n_f:n_f + n_p] = -sigma * psi * Pvec_low
        A.append(row); b.append(sigma * psi * P_lead)

    A = np.asarray(A)
    b = np.asarray(b)
    # Cap h from above: at strongly feasible levels (all Meff clamped to
    # 0) the LP would otherwise ride F to the coefficient box, which some
    # HiGHS builds report as unbounded.  Only h > tol matters to callers,
    # so the cap never changes a feasibility verdict.
    bounds = [(-coef_bound, coef_bound)] * (n_f + n_p) + [(None, 1e6)]
    return linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")


def _extract_poly(x_sol, nF, nP):
    """Return (F_cheb, P_cheb) with the paper's classical lc(P)=1
    normalisation (coefficient of T_nP in P is 1)."""
    n_f = nF + 1
    n_p = nP
    F = x_sol[:n_f].copy()
    if n_p:
        P = np.concatenate([x_sol[n_f:n_f + n_p], [1.0]])
    else:
        P = np.array([1.0])
    return F, P


def _cheb_eval(coeffs, x):
    """Evaluate sum_i c_i T_i(x) at x (scalar or ndarray)."""
    return npch.chebval(x, np.asarray(coeffs))


def _cheb_to_monomial(cheb_coeffs):
    """Convert Chebyshev coefficients to monomial coefficients in the same
    variable, ordered as [c_0, c_1, ..., c_n]."""
    return npch.cheb2poly(np.asarray(cheb_coeffs))


# -----------------------------------------------------------------------------
# Exchange-style feasibility probe with worst-violator adaptation
# -----------------------------------------------------------------------------

def _initial_samples(intervals, base_count=25):
    """Generate initial Chebyshev samples with density scaled by width."""
    pts = []
    for k, (a, b) in enumerate(intervals):
        width = abs(b - a)
        n = int(np.ceil(base_count * max(1.0, 0.5 + np.log10(1 + 10 * width))))
        pts.append((k, _chebyshev_nodes(a, b, n)))
    return pts


def _probe_lp(passbands, stopbands, sigma_I, sigma_J, nF, nP,
              M, psi_I_fn, psi_J_fn, F_ref=None,
              coef_bound=1e8, base_samples=25, refine_samples=2000,
              max_exchange=12, tol=1e-6, verbose=False):
    """Solve the LP at level M with the given reference F_ref (used in the
    quadratic-correction denominator of eq. 14).  A Remez-style exchange
    refines the sample sets until the continuous constraints are valid."""
    # Fixed-count sampling per interval, no width scaling.  Width-scaled
    # sampling over-constrains wide outer stopbands relative to narrow
    # middle stopbands, biasing the LP's rejection allocation and
    # converging to unbalanced solutions (lunot_gpt5.5 makes the same
    # observation implicitly by using a flat `points_per_interval`).
    n = max(6, base_samples)
    I_samples_set = []
    for k, (a, b) in enumerate(passbands):
        for x in _chebyshev_nodes(a, b, n):
            I_samples_set.append((x, sigma_I[k]))
    J_samples_set = []
    for k, (a, b) in enumerate(stopbands):
        for y in _chebyshev_nodes(a, b, n):
            J_samples_set.append((y, sigma_J[k]))

    F = P = None
    h = -np.inf

    for it in range(max_exchange):
        res = _build_lp(I_samples_set, J_samples_set,
                        nF, nP, M, psi_I_fn, psi_J_fn, coef_bound,
                        F_ref=F_ref)
        if not res.success:
            # A re-solve can fail numerically mid-exchange.  Return the
            # last successful iterate WITH its h: since the exchange only
            # adds constraints, the true exchanged value is <= h, so
            # h <= tol still certifies infeasibility, while h > tol
            # correctly reports "feasible at the sampled subset" (callers
            # verify candidates densely anyway).  Only a first-solve
            # failure returns (h=-inf, None, None): no conclusion.
            if verbose:
                print(f"      LP fail at exchange {it}: {res.message}")
            return h, F, P

        h = res.x[-1]
        F, P = _extract_poly(res.x, nF, nP)

        new_pts_I = []
        new_pts_J = []

        for k, (a, b) in enumerate(passbands):
            xs = np.linspace(a, b, refine_samples)
            sig = sigma_I[k]
            psi_x = np.array([psi_I_fn(x) for x in xs])
            Fx = _cheb_eval(F, xs)
            Px = _cheb_eval(P, xs)
            # Ratio-relative violation: |P| is tiny on I relative to the
            # huge F values on J, so an absolute threshold lets the LP
            # cheat the return-loss constraint by percents between
            # samples.  Normalising by |P| measures the violation in the
            # natural |F/P|-vs-psi units of the problem.
            v_ratio = ((np.abs(Fx) - psi_x * np.abs(Px))
                       / np.maximum(np.abs(Px), 1e-12))
            v_sig = -sig * Px
            thr_ratio = 10.0 * tol * np.maximum(1.0, psi_x)
            worst = int(np.argmax(v_ratio - thr_ratio))
            if (v_ratio - thr_ratio)[worst] > 0:
                new_pts_I.append((xs[worst], sig))
            worst = int(np.argmax(v_sig))
            if v_sig[worst] > tol:
                new_pts_I.append((xs[worst], sig))

        for k, (a, b) in enumerate(stopbands):
            ys = np.linspace(a, b, refine_samples)
            sig = sigma_J[k]
            w_y = np.array([_stopband_weight(psi_J_fn(y)) for y in ys])
            Meff = np.maximum(w_y * M, 0.0)  # match the clamped LP rows
            Fy = _cheb_eval(F, ys)
            Py = _cheb_eval(P, ys)
            if F_ref is not None:
                rhs = np.abs(_cheb_eval(F_ref, ys))
            else:
                rhs = np.ones_like(ys)
            v = h * rhs - (sig * Fy - Meff * np.abs(Py))
            worst = int(np.argmax(v))
            if v[worst] > tol * max(1.0, abs(h)):
                new_pts_J.append((ys[worst], sig))

        if not new_pts_I and not new_pts_J:
            break

        if verbose:
            print(f"      exchange {it}: added {len(new_pts_I)} I, "
                  f"{len(new_pts_J)} J (h={h:.4e})")

        I_samples_set.extend(new_pts_I)
        J_samples_set.extend(new_pts_J)

    return h, F, P


def _probe_lp_robust(passbands, stopbands, sigma_I, sigma_J, nF, nP,
                     M, psi_I_fn, psi_J_fn, F_ref=None,
                     coef_bound=1e8, base_samples=25, refine_samples=2000,
                     max_exchange=12, tol=1e-6, verbose=False):
    """_probe_lp with a mirror retry.

    (sigma_I, sigma_J) and (sigma_I, -sigma_J) describe the SAME
    subproblem under F -> -F (the passband constraints |F| <= psi sigma P
    and sigma P >= 0 are invariant), but the LP is ill-conditioned --
    polynomial values span many orders of magnitude across the bands --
    and the two mirrored models take different numerical paths through
    the solver.  One mirror regularly succeeds where the other returns a
    bogus near-zero h or fails outright.  Whenever the first attempt does
    not come back clearly feasible we retry mirrored and keep the better
    outcome (mapping the witness back via F -> -F).  An infeasibility
    certificate therefore requires BOTH mirrors to agree.  |F_ref| is
    mirror-invariant, so the same reference works for both.
    """
    h, F, P = _probe_lp(passbands, stopbands, sigma_I, sigma_J, nF, nP,
                        M, psi_I_fn, psi_J_fn, F_ref=F_ref,
                        coef_bound=coef_bound, base_samples=base_samples,
                        refine_samples=refine_samples,
                        max_exchange=max_exchange, tol=tol, verbose=verbose)
    if h > tol:
        return h, F, P
    sJ_m = [-s for s in sigma_J]
    h2, F2, P2 = _probe_lp(passbands, stopbands, sigma_I, sJ_m, nF, nP,
                           M, psi_I_fn, psi_J_fn, F_ref=F_ref,
                           coef_bound=coef_bound, base_samples=base_samples,
                           refine_samples=refine_samples,
                           max_exchange=max_exchange, tol=tol, verbose=verbose)
    if h2 > h:
        return h2, (None if F2 is None else -F2), P2
    return h, F, P


def _repair_to_passband(F, P, passbands, sigma_I, psi_I_fn, n_samples):
    """Make a candidate honour |F| <= psi sigma P on the WHOLE of I, not
    just at the LP samples: an LP iterate (un-exchanged seed, or a probe
    whose exchange budget ran out) can cheat between passband samples
    while showing a large stopband slack.  Scaling F down by the worst
    dense ratio restores validity at a proportional cost in slack.

    Returns the repaired F (the original when already valid), or None
    when sigma P <= 0 somewhere on I (not repairable by scaling)."""
    ratio_max = 1.0
    for (a, b), sig in zip(passbands, sigma_I):
        xs = np.linspace(a, b, n_samples)
        Fx = _cheb_eval(F, xs)
        Px = _cheb_eval(P, xs)
        sP = sig * Px
        if np.any(sP <= 0.0):
            return None
        psi = np.array([psi_I_fn(x) for x in xs])
        ratio_max = max(ratio_max, float(np.max(np.abs(Fx) / (psi * sP))))
    if ratio_max <= 1.0:
        return np.asarray(F, dtype=float)
    return np.asarray(F, dtype=float) / (ratio_max * (1.0 + 1e-12))


def _min_signed_ratio_minus_psi(F, P, stopbands, sigma_J, psi_J_fn,
                                n_samples=6000):
    """min_{y in J}  sigma(y) F(y) / (|P(y)| w_J(y)).

    The dense (continuum-verified) value of the multiplicatively
    weighted criterion.  When the candidate (F, P) has F with the
    expected sign on each J interval this equals min |F/P| / w_J; a
    wrong sign configuration produces a large negative value, which
    prunes that sign combination from the outer search."""
    vals = []
    for (a, b), sig in zip(stopbands, sigma_J):
        ys = np.linspace(a, b, n_samples)
        Fy = _cheb_eval(F, ys)
        Py = _cheb_eval(P, ys)
        good = np.abs(Py) > 1e-14
        r = sig * Fy[good] / np.abs(Py[good])
        w = np.array([_stopband_weight(psi_J_fn(y)) for y in ys[good]])
        vals.append(r / w)
    return float(np.min(np.concatenate(vals))) if vals else np.inf


def _probe_feasibility(passbands, stopbands, sigma_I, sigma_J, nF, nP,
                       M, psi_I_fn, psi_J_fn,
                       coef_bound=1e8, base_samples=25, refine_samples=2000,
                       max_exchange=10, tol=1e-6, verbose=False):
    """Kept for bisection compatibility: returns (feasible, h, F, P)
    assuming F_ref = 1 (non-quadratic form)."""
    h, F, P = _probe_lp(passbands, stopbands, sigma_I, sigma_J, nF, nP,
                       M, psi_I_fn, psi_J_fn, F_ref=None,
                       coef_bound=coef_bound, base_samples=base_samples,
                       refine_samples=refine_samples,
                       max_exchange=max_exchange, tol=tol, verbose=verbose)
    return (h > -tol), h, F, P


# -----------------------------------------------------------------------------
# Paper's iterative differential correction, eq (10)/(14) -- quadratic conv.
# -----------------------------------------------------------------------------

def _dc_iterate(passbands, stopbands, sigma_I, sigma_J, nF, nP,
                psi_I_fn, psi_J_fn, F0, P0, M0,
                coef_bound, base_samples, refine_samples,
                max_iter, tol, verbose=False):
    """Eq. (14) quadratic-correction loop from a given seed (F0, P0, M0).

        Step k:  solve LP at M = M_{k-1} with F_ref = F_{k-1}.
                 If the LP's h <= 0 return the best iterate so far.
                 Else  M_k = min_J sigma F_k / (|P_k| w_J)  (dense).
    """
    F_km1, P_km1, M_km1 = F0, P0, M0
    best_F, best_P, best_M = None, None, -np.inf

    def consider(F, P):
        """Repair the candidate to dense passband validity, re-measure
        its slack, and keep it when it beats the current best."""
        nonlocal best_F, best_P, best_M
        F_rep = _repair_to_passband(F, P, passbands, sigma_I, psi_I_fn,
                                    refine_samples)
        if F_rep is None:
            return
        M_rep = _min_signed_ratio_minus_psi(F_rep, P, stopbands,
                                            sigma_J, psi_J_fn,
                                            n_samples=refine_samples)
        if M_rep > best_M:
            best_F, best_P, best_M = F_rep, P, M_rep

    consider(F0, P0)
    for k in range(1, max_iter + 1):
        h, F_k, P_k = _probe_lp_robust(
            passbands, stopbands, sigma_I, sigma_J, nF, nP,
            M=M_km1, psi_I_fn=psi_I_fn, psi_J_fn=psi_J_fn, F_ref=F_km1,
            coef_bound=coef_bound, base_samples=base_samples,
            refine_samples=refine_samples, tol=tol, verbose=False,
        )
        if F_k is None or h <= tol:
            if verbose:
                print(f"    iter {k}: stop (h={h:.3e})")
            break
        M_k = _min_signed_ratio_minus_psi(F_k, P_k, stopbands,
                                          sigma_J, psi_J_fn,
                                          n_samples=refine_samples)
        if verbose:
            print(f"    iter {k}: h={h:.3e}  M={M_k:.5f}  dM={M_k-M_km1:.2e}")
        consider(F_k, P_k)
        if M_k - M_km1 < tol * max(1.0, abs(M_k)):
            break
        F_km1, P_km1, M_km1 = F_k, P_k, M_k
    return best_F, best_P, best_M


def _solve_signed_diffcorr(passbands, stopbands, sigma_I, sigma_J, nF, nP,
                           psi_I_fn, psi_J_fn,
                           coef_bound=1e8, base_samples=25,
                           refine_samples=3000,
                           max_iter=30, tol=1e-6,
                           prune_below=-np.inf, kick_budget=24,
                           verbose=False):
    """Paper's Section V.C algorithm with the eq. (14) quadratic correction.

        Step 0:  solve LP at M_{-1}=1, F_{-1}=1 to get (F_0, P_0).
                 Compute M_0 = min_J sigma F_0 / (|P_0| w_J).
        Step k:  solve LP at M = M_{k-1} with F_ref = F_{k-1}.
                 If the LP's h <= 0 return (F_{k-1}, P_{k-1}, M_{k-1}).
                 Else  M_k = min_J sigma F_k / (|P_k| w_J).

    The `min` is computed densely (verification of the continuum).
    The DC loop is followed by a feasibility-kick refinement (see below)
    that escapes DC stalls and produces a certified upper bound.

    Returns (F, P, M, M_upper):  M is dense-verified achieved slack,
    M_upper a certified bound the signed optimum cannot exceed (np.inf if
    the kick budget ran out before an infeasible probe was found).
    """
    # Step 0: bootstrap (F_0, P_0) by trying a sequence of initial M
    # values that bracket the likely optimum from above.  For each M in
    # the sequence we try the basic (non-quadratic) LP; the *first* M
    # yielding h > tol becomes the seed.
    # In the multiplicative weighting any level < 0 clamps every
    # stopband row to the vacuous sign-pinning form, so a single
    # negative fallback suffices (it is always feasible whenever the
    # sign pattern admits a candidate at all).
    initial_margins = (1.0, 0.3, 0.1, 0.03, 0.01, 0.0, -1.0)

    # IMPORTANT: we disable the Remez exchange during initialisation.
    # Starting from the Chebyshev-Lobatto samples, the first LP solve
    # already gives a usable seed; adding violating points at this stage
    # only tightens the problem until the LP becomes infeasible at the
    # very margins we're trying to bracket.  The iterative DC below
    # re-solves with F_ref updated, which effectively plays the exchange
    # role for convergence.
    h0, F_km1, P_km1, M_start = None, None, None, None
    for trial_M in initial_margins:
        h_try, F_try, P_try = _probe_lp_robust(
            passbands, stopbands, sigma_I, sigma_J, nF, nP,
            M=trial_M, psi_I_fn=psi_I_fn, psi_J_fn=psi_J_fn, F_ref=None,
            coef_bound=coef_bound, base_samples=base_samples,
            refine_samples=refine_samples, max_exchange=1, tol=tol,
            verbose=False,
        )
        if F_try is None:
            continue
        if h_try > tol or trial_M < 0:
            h0, F_km1, P_km1, M_start = h_try, F_try, P_try, trial_M
            if h_try > tol:
                break
    if F_km1 is None:
        # The M-independent passband constraints (sigma P >= 0,
        # |F| <= psi sigma P) admit no candidate for this sign pattern.
        return None, None, -np.inf, -np.inf
    M_km1 = _min_signed_ratio_minus_psi(F_km1, P_km1, stopbands,
                                        sigma_J, psi_J_fn,
                                        n_samples=refine_samples)
    # The un-exchanged seed cheats between its samples, overstating its
    # slack; starting DC at the overstated level makes the very first
    # probe infeasible and kills the iteration.  Start from the honest
    # (repaired) level instead.
    F0_rep = _repair_to_passband(F_km1, P_km1, passbands, sigma_I,
                                 psi_I_fn, refine_samples)
    if F0_rep is not None:
        M_km1 = _min_signed_ratio_minus_psi(F0_rep, P_km1, stopbands,
                                            sigma_J, psi_J_fn,
                                            n_samples=refine_samples)
    if verbose:
        print(f"    init:  M_start={M_start:.3f} h0={h0:.3e}  M_0={M_km1:.4f}")

    best_F, best_P, best_M = _dc_iterate(
        passbands, stopbands, sigma_I, sigma_J, nF, nP,
        psi_I_fn, psi_J_fn, F_km1, P_km1, M_km1,
        coef_bound, base_samples, refine_samples, max_iter, tol,
        verbose=verbose)

    # ------------------------------------------------------------------
    # Feasibility-kick refinement (global escape + optimality bracket).
    #
    # The DC iteration above is fast but only locally reliable: the LP has
    # massively degenerate optimal faces, and which vertex the backend
    # happens to return decides which valley DC follows.  (Observed: the
    # same Example-2 sign combination converging to M=1.62 with one HiGHS
    # build and M=15.16 with another.)  The *feasibility test* at a fixed
    # level L -- "does a candidate with margin > L exist?" -- is a single
    # LP with no such ambiguity:
    #
    #     feasible (h > 0)  =>  a strictly better candidate exists: climb;
    #     infeasible        =>  certificate that no candidate beats L,
    #                           even on the sampled problem, hence
    #                           certainly not on the continuum.
    #
    # We probe at L = floor + kick (with the mirror retry of
    # _probe_lp_robust, so a certificate requires both mirrors to agree).
    # On feasibility we adopt the witness candidate, let DC polish it
    # quadratically, and double the kick so a long climb costs O(log)
    # probes.  On infeasibility L is a certified upper bound and the
    # kick halves down to a ~1% gap.  `prune_below` lets the caller pass
    # the best M achieved by other sign combinations: a combination that
    # cannot beat it is abandoned after a single infeasible probe.
    # ------------------------------------------------------------------
    def kick0(m):
        return max(0.05 * (1.0 + abs(m)), 10.0 * tol)

    def kick_min(m):
        return max(0.01 * (1.0 + abs(m)), 10.0 * tol)

    M_upper = np.inf
    # When no dense-valid candidate emerged from DC, fall back to the
    # raw init seed for the probe level.
    floor = max(best_M, prune_below)
    if not np.isfinite(floor):
        floor = M_km1
    kick = kick0(floor)
    stalls = 0
    for _ in range(max(0, kick_budget)):
        L = floor + kick
        # Probe ladder: both normalisations of h (relative units with
        # F_ref = current best F, the eq. 14 denominator, and absolute
        # units rhs = 1) x two Lobatto grid densities (x the two sigma_J
        # mirrors inside _probe_lp_robust).  Each axis decorrelates a
        # failure mode that has been observed to produce a false
        # "infeasible" on this ill-conditioned LP: the h normalisation
        # changes the objective scaling, the grid density changes the
        # near-dependent row structure.  The first clearly feasible
        # witness wins; a certificate requires every rung to agree.
        F_ref_kick = best_F if best_F is not None else F_km1
        bs_alt = max(12, (2 * base_samples) // 3)
        h_t, F_t, P_t = -np.inf, None, None
        for F_ref_v, bs_v in ((F_ref_kick, base_samples),
                              (None, base_samples),
                              (F_ref_kick, bs_alt),
                              (None, bs_alt)):
            h_v, F_v, P_v = _probe_lp_robust(
                passbands, stopbands, sigma_I, sigma_J, nF, nP,
                M=L, psi_I_fn=psi_I_fn, psi_J_fn=psi_J_fn, F_ref=F_ref_v,
                coef_bound=coef_bound, base_samples=bs_v,
                refine_samples=refine_samples, tol=tol, verbose=False)
            if F_v is not None and (F_t is None or h_v > h_t):
                h_t, F_t, P_t = h_v, F_v, P_v
            if F_t is not None and h_t > tol:
                break
        if F_t is not None and h_t > tol:
            # A strictly better candidate exists; measure it on the
            # continuum and resume quadratic DC from it.
            M_t = _min_signed_ratio_minus_psi(F_t, P_t, stopbands,
                                              sigma_J, psi_J_fn,
                                              n_samples=refine_samples)
            F_r, P_r, M_r = _dc_iterate(
                passbands, stopbands, sigma_I, sigma_J, nF, nP,
                psi_I_fn, psi_J_fn, F_t, P_t, M_t,
                coef_bound, base_samples, refine_samples, max_iter, tol,
                verbose=verbose)
            if M_r > best_M:
                best_F, best_P, best_M = F_r, P_r, M_r
            if best_M > M_upper:
                # The climb just contradicted an earlier "certificate":
                # that bound was solver noise.  Discard it.
                M_upper = np.inf
            if verbose:
                print(f"    kick L={L:.5f}: feasible, climbed to {best_M:.5f}")
            # Geometric growth: as long as probes keep coming back
            # feasible, double the kick so a long climb costs O(log)
            # probes instead of O(1/kick).  (DC usually overshoots L
            # anyway; the doubling matters when it keeps stalling.)
            new_floor = max(best_M, prune_below)
            if not np.isfinite(new_floor):
                new_floor = L
            # Repeated feasible probes without any dense improvement mean
            # the witnesses cannot be realised on the continuum at this
            # resolution -- stop instead of oscillating.
            stalls = stalls + 1 if new_floor <= floor + tol else 0
            if stalls >= 3:
                break
            floor = new_floor
            kick = max(2.0 * kick, kick0(floor))
        else:
            if F_t is not None:
                # Genuine h <= tol: certified -- no candidate beats L.
                M_upper = min(M_upper, L)
                if verbose:
                    print(f"    kick L={L:.5f}: infeasible (upper bound)")
            elif verbose:
                # LP failure on the very first solve: no certificate at
                # this level; retry lower.
                print(f"    kick L={L:.5f}: LP failed (no certificate)")
            if kick <= kick_min(floor) * (1.0 + 1e-9):
                break
            kick = max(0.5 * kick, kick_min(floor))

    # A bound below the dense-verified achievement is solver noise.
    M_upper = max(M_upper, best_M)
    return best_F, best_P, best_M, M_upper


# -----------------------------------------------------------------------------
# Bisection on M for a fixed sign combination
# -----------------------------------------------------------------------------

def _solve_signed_bisection(passbands, stopbands, sigma_I, sigma_J, nF, nP,
                            psi_I_fn, psi_J_fn,
                            M_lo=None, M_hi=None,
                            bisection_tol=1e-5,
                            coef_bound=1e8, base_samples=25,
                            refine_samples=2000, verbose=False):
    """Bisect on M to find the largest level at which the LP is feasible
    (h > 0), for the given sign combination.  If M_lo is not given we
    search downward until the LP becomes feasible (allowing M < 0,
    meaning the optimal filter undershoots the psi specification).

    Returns (F, P, M, M_upper); M_upper is the lowest level probed
    infeasible (a certified bound on the signed optimum)."""
    # Find a feasible lower bracket.  Try M=0 first; if that is infeasible
    # descend geometrically in |M|.
    if M_lo is None:
        M_lo = 0.0
        feas_lo, _, F_best, P_best = _probe_feasibility(
            passbands, stopbands, sigma_I, sigma_J, nF, nP,
            M_lo, psi_I_fn, psi_J_fn,
            coef_bound=coef_bound, base_samples=base_samples,
            refine_samples=refine_samples, verbose=False)
        step = -1.0
        while not feas_lo:
            M_lo += step
            step *= 2.0
            feas_lo, _, F_best, P_best = _probe_feasibility(
                passbands, stopbands, sigma_I, sigma_J, nF, nP,
                M_lo, psi_I_fn, psi_J_fn,
                coef_bound=coef_bound, base_samples=base_samples,
                refine_samples=refine_samples, verbose=False)
            if M_lo < -1e4:
                if verbose:
                    print(f"    giving up lo-bracket at M_lo={M_lo}")
                return None, None, -np.inf, -np.inf
            if verbose:
                print(f"    probing M_lo={M_lo:.2f} feas={feas_lo}")
    else:
        feas_lo, _, F_best, P_best = _probe_feasibility(
            passbands, stopbands, sigma_I, sigma_J, nF, nP,
            M_lo, psi_I_fn, psi_J_fn,
            coef_bound=coef_bound, base_samples=base_samples,
            refine_samples=refine_samples, verbose=False)
        if not feas_lo:
            if verbose:
                print(f"    low bound M={M_lo} already infeasible.")
            return None, None, -np.inf, -np.inf

    # Grow M_hi until infeasible.
    if M_hi is None:
        M_hi = max(M_lo + 1.0, 1.0)
        for _ in range(40):
            feas, h, F, P = _probe_feasibility(
                passbands, stopbands, sigma_I, sigma_J, nF, nP,
                M_hi, psi_I_fn, psi_J_fn,
                coef_bound=coef_bound, base_samples=base_samples,
                refine_samples=refine_samples, verbose=False)
            if verbose:
                print(f"    hi probe M={M_hi:.4g}: feas={feas} h={h:.3e}")
            if not feas:
                break
            if M_hi > 0:
                M_hi *= 2.0
            else:
                M_hi = M_hi / 2.0 + 1.0   # push through 0 quickly
        else:
            raise RuntimeError("Could not bracket M: feasible up to a huge "
                               "value - probably an unbounded/degenerate "
                               "formulation.")

    best_M = M_lo
    # Standard bisection.
    for it in range(60):
        if M_hi - M_lo <= bisection_tol * max(1.0, abs(M_lo)):
            break
        M_mid = 0.5 * (M_lo + M_hi)
        feas, h, F, P = _probe_feasibility(
            passbands, stopbands, sigma_I, sigma_J, nF, nP,
            M_mid, psi_I_fn, psi_J_fn,
            coef_bound=coef_bound, base_samples=base_samples,
            refine_samples=refine_samples, verbose=False)
        if verbose:
            print(f"    it {it}: M_mid={M_mid:.6f}  feas={feas}  h={h:.3e}")
        if feas:
            M_lo = M_mid
            F_best, P_best = F, P
            best_M = M_mid
        else:
            M_hi = M_mid

    return F_best, P_best, best_M, M_hi


# -----------------------------------------------------------------------------
# Top-level: enumerate sign combinations
# -----------------------------------------------------------------------------

def solve_zolotarev(passbands, stopbands, nF, nP,
                    psi_I=1.0, psi_J=0.0,
                    coef_bound=1e8, base_samples=30,
                    refine_samples=4000, bisection_tol=1e-5,
                    rescale=True, method="diffcorr", verbose=False):
    """
    Compute the certified optimal real filtering function (F, P) for the
    multiband specification.  Returns a dict with keys

        F, P           : polynomial coefficients in ascending order,
                         expressed in the ORIGINAL frequency variable.
        M              : best achieved weighted criterion
                         min_J |F/P| / w_J  (dense-verified on the
                         continuum).  With per-band dB targets,
                         20 log10(M) is the uniform margin in dB by
                         which every stopband target is exceeded
                         (negative: missed); with psi_J = 0 it is the
                         classical min |F/P|.
        M_upper        : certified upper bound -- no candidate of this
                         degree beats it even on the sampled problem
                         (max over the per-sign infeasibility certificates).
        sigma_I, sigma_J : the winning sign combination.
        passbands, stopbands, nF, nP, psi_I, psi_J : stored for plotting.
    """
    psi_I_fn_orig = _as_fn(psi_I)
    psi_J_fn_orig = _as_fn(psi_J)

    # Rescale omega to be comfortably inside [-1, 1].  This is purely a
    # numerical conditioning step; nothing of the geometry changes.
    if rescale:
        all_edges = []
        for (a, b) in passbands + stopbands:
            all_edges.extend([abs(a), abs(b)])
        scale = max(all_edges)
    else:
        scale = 1.0

    pb_s = [(a / scale, b / scale) for (a, b) in passbands]
    sb_s = [(a / scale, b / scale) for (a, b) in stopbands]
    psi_I_fn = lambda w, fn=psi_I_fn_orig: fn(w * scale)
    psi_J_fn = lambda w, fn=psi_J_fn_orig: fn(w * scale)

    r = len(pb_s)
    p = len(sb_s)
    best = None
    best_M = -np.inf
    global_upper = -np.inf

    for bits_I in product([1, -1], repeat=r):
        if bits_I[0] != 1:
            continue  # (F,P) -> (-F,-P) symmetry
        for bits_J in product([1, -1], repeat=p):
            if bits_J[0] != 1:
                # F -> -F (P unchanged) flips every sigma_J while leaving
                # the passband constraints |F| <= psi sigma_I P and
                # sigma_I P >= 0 invariant, so (sigma_I, sigma_J) and
                # (sigma_I, -sigma_J) are the same subproblem.
                continue
            if verbose:
                print(f"sigma_I={bits_I}  sigma_J={bits_J}")
            if method == "diffcorr":
                F_s, P_s, M_val, M_ub = _solve_signed_diffcorr(
                    pb_s, sb_s, list(bits_I), list(bits_J),
                    nF, nP, psi_I_fn, psi_J_fn,
                    coef_bound=coef_bound, base_samples=base_samples,
                    refine_samples=refine_samples,
                    max_iter=30, tol=bisection_tol,
                    prune_below=best_M, verbose=verbose)
            else:
                F_s, P_s, M_val, M_ub = _solve_signed_bisection(
                    pb_s, sb_s, list(bits_I), list(bits_J),
                    nF, nP, psi_I_fn, psi_J_fn,
                    M_lo=None, M_hi=None,
                    coef_bound=coef_bound, base_samples=base_samples,
                    refine_samples=refine_samples,
                    bisection_tol=bisection_tol, verbose=verbose)
            global_upper = max(global_upper, M_ub)
            if F_s is None:
                continue
            if verbose:
                print(f"  -> M={M_val:.5f}  (<= {M_ub:.5f})")
            if M_val > best_M:
                best_M = M_val
                best = (F_s, P_s, list(bits_I), list(bits_J))

    if best is None:
        raise RuntimeError("No sign combination yielded a feasible filter.")

    F_s, P_s, sI, sJ = best
    # F_s, P_s are CHEBYSHEV coefficients in t = omega / scale.  Convert
    # them to monomial coefficients in t, then rescale to monomial
    # coefficients in omega.
    F_mon_t = _cheb_to_monomial(F_s)
    P_mon_t = _cheb_to_monomial(P_s)
    F = _rescale_poly_coeffs(F_mon_t, 1.0 / scale)
    P = _rescale_poly_coeffs(P_mon_t, 1.0 / scale)

    return {
        "F": F, "P": P, "M": best_M,
        "M_upper": global_upper,
        "sigma_I": sI, "sigma_J": sJ,
        "passbands": passbands, "stopbands": stopbands,
        "nF": nF, "nP": nP, "psi_I": psi_I, "psi_J": psi_J,
        "scale": scale, "F_cheb_scaled": F_s, "P_cheb_scaled": P_s,
    }


def _rescale_poly_coeffs(coeffs, inv_scale):
    """
    If q(t) = sum c_i t^i with t = omega/scale, then q(omega/scale)
    = sum c_i (omega/scale)^i = sum (c_i / scale^i) omega^i, so the
    coefficients in omega are c_i * (1/scale)^i.
    """
    out = np.array(coeffs, dtype=float)
    return np.array([out[i] * (inv_scale ** i) for i in range(len(out))])


# -----------------------------------------------------------------------------
# Scattering-parameter evaluation and alternation-point certificate
# -----------------------------------------------------------------------------

def scattering_response(F, P, omega):
    """|S11|^2 and |S21|^2 from the filtering function D = F/P."""
    omega = np.asarray(omega, dtype=float)
    D = _poly_eval(F, omega) / _poly_eval(P, omega)
    D2 = D * D
    return D2 / (1.0 + D2), 1.0 / (1.0 + D2)


def poly_eval(coeffs, x):
    return _poly_eval(coeffs, x)


def extreme_points(F, P, passbands, stopbands, sigma_I, sigma_J,
                   psi_I=1.0, psi_J=0.0, M=None, tol=1e-2, n_dense=4000):
    """Approximate set of alternation points.  Returns list of
    (omega, sign, band_type)."""
    psi_I_fn = _as_fn(psi_I)
    psi_J_fn = _as_fn(psi_J)
    if M is None:
        # best-effort estimate of the weighted criterion
        vals = []
        for (a, b), sig in zip(stopbands, sigma_J):
            ws = np.linspace(a, b, n_dense)
            D = _poly_eval(F, ws) / _poly_eval(P, ws)
            wgt = np.array([_stopband_weight(psi_J_fn(w)) for w in ws])
            vals.append(np.abs(D) / wgt)
        M = float(np.min(np.concatenate(vals)))

    pts = []
    for (a, b), sig in zip(passbands, sigma_I):
        ws = np.linspace(a, b, n_dense)
        D = _poly_eval(F, ws) / _poly_eval(P, ws)
        psi = np.array([psi_I_fn(w) for w in ws])
        for resid, which in ((D - psi, +1), (D + psi, -1)):
            abs_r = np.abs(resid)
            for i in range(1, len(abs_r) - 1):
                if abs_r[i] <= abs_r[i - 1] and abs_r[i] <= abs_r[i + 1]:
                    if abs_r[i] < tol:
                        pts.append((ws[i], which, "I"))

    for (a, b), sig in zip(stopbands, sigma_J):
        ws = np.linspace(a, b, n_dense)
        D = _poly_eval(F, ws) / _poly_eval(P, ws)
        wgt = np.array([_stopband_weight(psi_J_fn(w)) for w in ws])
        target = M * wgt
        if sig == +1:
            resid = D - target
            which = -1
        else:
            resid = D + target
            which = +1
        abs_r = np.abs(resid)
        for i in range(1, len(abs_r) - 1):
            if abs_r[i] <= abs_r[i - 1] and abs_r[i] <= abs_r[i + 1]:
                if abs_r[i] < tol * max(1.0, abs(target[i])):
                    pts.append((ws[i], which, "J"))

    pts.sort(key=lambda t: t[0])
    deduped = []
    for w, s, t in pts:
        if deduped and abs(w - deduped[-1][0]) < 1e-4:
            continue
        deduped.append((w, s, t))
    return deduped


def alternation_length(ext_pts):
    """Length of the longest alternating subsequence of signs."""
    if not ext_pts:
        return 0
    run = 1
    best = 1
    prev = ext_pts[0][1]
    for (_, s, _) in ext_pts[1:]:
        if s != prev:
            run += 1
            prev = s
            best = max(best, run)
        else:
            run = 1
            prev = s
    return best


# -----------------------------------------------------------------------------
# dB helpers
# -----------------------------------------------------------------------------

def rl_db_to_psi(rl_db: float) -> float:
    """Passband return loss (dB) -> max |F/P| in passband."""
    r = 10.0 ** (-rl_db / 20.0)
    return r / np.sqrt(1.0 - r * r)


def rej_db_to_psi(rej_db: float) -> float:
    """Stopband rejection level (dB) -> baseline |F/P|."""
    return float(np.sqrt(10.0 ** (rej_db / 10.0) - 1.0))


# -----------------------------------------------------------------------------
# E(s) denominator: Feldtkeller spectral factorisation
# -----------------------------------------------------------------------------
#
# From eq. (2) of the paper, E is the Hurwitz polynomial (all roots in the
# open LHP of s = sigma + j*omega) satisfying
#
#     E(s) E*(s) = F(s) F*(s) + (-1)^(n+1) P(s)^2
#
# For real-coefficient F, P on the real omega axis with the substitution
# s = j*omega, this reduces to
#
#     |E(j*omega)|^2 = F(omega)^2 + P(omega)^2 .
#
# Roots of E are the filter poles: the main downstream handoff to
# coupling-matrix synthesis / physical CAD.


def compute_E_polynomial(F_mono, P_mono):
    """Compute the monic Hurwitz E(s) such that |E(j*omega)|^2 = F^2 + P^2.

    Parameters
    ----------
    F_mono, P_mono : monomial coefficients in ascending order, real.

    Returns
    -------
    dict with keys
        E_s_coeffs : complex coefficients of E(s) (ascending order, deg nF).
        poles      : array of E's roots (complex), sorted by |Im|.
        residual   : relative L2 error of the spectral reconstruction
                     (should be ~1e-10 if all is well).
    """
    F = np.trim_zeros(np.asarray(F_mono, dtype=float), 'b')
    P = np.trim_zeros(np.asarray(P_mono, dtype=float), 'b')
    if F.size == 0:
        raise ValueError("F is zero")
    nF = F.size - 1

    # |E(j*omega)|^2 = F(omega)^2 + P(omega)^2, as a polynomial of omega.
    FF = np.convolve(F, F)                          # F(omega)^2
    PP = np.convolve(P, P)                          # P(omega)^2
    L = max(FF.size, PP.size)
    T = np.zeros(L)
    T[:FF.size] += FF
    T[:PP.size] += PP

    # T(omega) is >= 0 on the real axis with deg 2*nF.  Find its 2*nF complex
    # roots, then convert each root  omega_k  to  s_k = j*omega_k.  The s-
    # roots come in pairs (s, -conj(s)); E's roots are the ones with
    # Re(s) < 0.
    roots_omega = np.roots(T[::-1])
    roots_s = 1j * roots_omega

    lhp = [r for r in roots_s if r.real < -1e-12]
    if len(lhp) != nF:
        # Tolerance case: some roots may sit on the j-axis (T vanishing at a
        # real point because F and P share a zero).  Split them off by
        # preferring those with negative real part.
        roots_sorted = sorted(roots_s, key=lambda r: r.real)
        lhp = roots_sorted[:nF]

    # Build E from its roots (LHP) as a monic polynomial first.
    E = np.poly(lhp)[::-1].astype(np.complex128)   # ascending order, monic

    # Spectral match: |E(j*omega)|^2 must equal T(omega) pointwise.  Since
    # both sides are degree-2nF polynomials of omega^2, matching the
    # leading coefficient is enough.  deg(T) = 2 nF with leading coefficient
    # = F_lead^2 (P has degree < nF, so it's sub-leading), and a monic E
    # gives leading |E(j omega)|^2 = omega^(2 nF).  Hence we multiply E by
    # |F_lead| to match.
    F_lead = float(F[-1])
    E = E * abs(F_lead)

    # Verification on a dense grid spanning the band edges.
    if F_lead != 0:
        ws = np.linspace(-2.0, 2.0, 401)
        Ej = np.polyval(E[::-1], 1j * ws)
        lhs = (Ej * np.conjugate(Ej)).real
        rhs = np.polyval(T[::-1], ws)
        residual = float(np.linalg.norm(lhs - rhs)
                         / max(np.linalg.norm(rhs), 1e-30))
    else:
        residual = float('nan')

    return {
        "E_s_coeffs": E,
        "poles": np.array(sorted(lhp, key=lambda r: (abs(r.imag), r.real))),
        "residual": residual,
    }
