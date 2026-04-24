"""
Certified multiband filter synthesis after

    V. Lunot, F. Seyfert, S. Bila, A. Nasser,
    "Certified computation of optimal multiband filtering functions,"
    IEEE Trans. MTT, 56(1), pp. 105-112, 2008.

The real Zolotarev problem:

    given passband intervals I = U I_k  and stopband intervals J = U J_k,
    find real-coefficient polynomials F, P with deg F = nF, deg P <= nP
    maximising   min_{w in J}  |F(w)/P(w)|
    subject to   |F(w)/P(w)|  <=  psi_I(w)  on I,
                 |F(w)/P(w)|  >=  M + psi_J(w)  on J   (M maximised).

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
        s.t.   sigma F(y) - (psi + M) |P(y)| >= h |F_ref(y)|   for y in J
               sigma P(x) >= 0                                 for x in I
               |F(x)| <= sigma psi(x) P(x)                     for x in I

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
        psi = psi_J_fn(y)
        Meff = psi + M
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
    bounds = [(-coef_bound, coef_bound)] * (n_f + n_p) + [(None, None)]
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
            if verbose:
                print(f"      LP fail at exchange {it}: {res.message}")
            return -np.inf, F, P

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
            v_ratio = np.abs(Fx) - psi_x * np.abs(Px)
            v_sig   = -sig * Px
            for v in (v_ratio, v_sig):
                worst = int(np.argmax(v))
                if v[worst] > tol:
                    new_pts_I.append((xs[worst], sig))

        for k, (a, b) in enumerate(stopbands):
            ys = np.linspace(a, b, refine_samples)
            sig = sigma_J[k]
            psi_y = np.array([psi_J_fn(y) for y in ys])
            Meff = psi_y + M
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


def _min_signed_ratio_minus_psi(F, P, stopbands, sigma_J, psi_J_fn,
                                n_samples=6000):
    """min_{y in J}  (sigma(y) F(y) / |P(y)| - psi_J(y)).

    When the candidate (F, P) has F with the expected sign on each J
    interval, this equals  min_{y in J} (|F/P| - psi_J).  A wrong sign
    configuration produces a large negative value, which prunes that
    sign combination from the outer search."""
    vals = []
    for (a, b), sig in zip(stopbands, sigma_J):
        ys = np.linspace(a, b, n_samples)
        Fy = _cheb_eval(F, ys)
        Py = _cheb_eval(P, ys)
        good = np.abs(Py) > 1e-14
        r = sig * Fy[good] / np.abs(Py[good])
        psi = np.array([psi_J_fn(y) for y in ys[good]])
        vals.append(r - psi)
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

def _solve_signed_diffcorr(passbands, stopbands, sigma_I, sigma_J, nF, nP,
                           psi_I_fn, psi_J_fn,
                           coef_bound=1e8, base_samples=25,
                           refine_samples=3000,
                           max_iter=30, tol=1e-6, verbose=False):
    """Paper's Section V.C algorithm with the eq. (14) quadratic correction.

        Step 0:  solve LP at M_{-1}=1, F_{-1}=1 to get (F_0, P_0).
                 Compute M_0 = min_J (sigma F_0 / |P_0| - psi_J).
        Step k:  solve LP at M = M_{k-1} with F_ref = F_{k-1}.
                 If the LP's h <= 0 return (F_{k-1}, P_{k-1}, M_{k-1}).
                 Else  M_k = min_J (sigma F_k / |P_k| - psi_J).

    The `min` is computed densely (verification of the continuum).
    """
    # Step 0: bootstrap (F_0, P_0) by trying a sequence of initial M values
    # that bracket the likely optimum from above.  This is the key trick
    # the gpt5.5 port uses to escape the poor local optima we were hitting
    # with a single safe-but-loose M_start = -psi_max-1.  For each M in
    # the sequence we try the basic (non-quadratic) LP; the *first* M
    # yielding h > tol becomes the seed.  Only if all positive margins
    # fail do we fall back to the always-feasible M = -psi_max - 1.
    psi_J_max = 0.0
    for (a, b) in stopbands:
        for w in np.linspace(a, b, 11):
            psi_J_max = max(psi_J_max, psi_J_fn(w))
    initial_margins = (1.0, 0.3, 0.1, 0.03, 0.01, 0.0,
                       -0.5 * max(psi_J_max, 1.0),
                       -psi_J_max - 1.0)

    # IMPORTANT: we disable the Remez exchange during initialisation.
    # Starting from the Chebyshev-Lobatto samples, the first LP solve
    # already gives a usable seed; adding violating points at this stage
    # only tightens the problem until the LP becomes infeasible at the
    # very margins we're trying to bracket.  The iterative DC below
    # re-solves with F_ref updated, which effectively plays the exchange
    # role for convergence.
    h0, F_km1, P_km1, M_start = None, None, None, None
    for trial_M in initial_margins:
        h_try, F_try, P_try = _probe_lp(
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
        return None, None, -np.inf
    M_km1 = _min_signed_ratio_minus_psi(F_km1, P_km1, stopbands,
                                        sigma_J, psi_J_fn,
                                        n_samples=refine_samples)
    if verbose:
        print(f"    init:  M_start={M_start:.3f} h0={h0:.3e}  M_0={M_km1:.4f}")

    best_F, best_P, best_M = F_km1, P_km1, M_km1

    for k in range(1, max_iter + 1):
        h, F_k, P_k = _probe_lp(
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
        if M_k > best_M:
            best_M = M_k
            best_F, best_P = F_k, P_k
        if M_k - M_km1 < tol * max(1.0, abs(M_k)):
            break
        F_km1, P_km1, M_km1 = F_k, P_k, M_k

    return best_F, best_P, best_M


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
    meaning the optimal filter undershoots the psi specification)."""
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
                return None, None, -np.inf
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
            return None, None, -np.inf

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

    return F_best, P_best, best_M


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
        M              : best achieved slack above psi_J on J.
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

    for bits_I in product([1, -1], repeat=r):
        if bits_I[0] != 1:
            continue  # (F,P) -> (-F,-P) symmetry
        for bits_J in product([1, -1], repeat=p):
            if verbose:
                print(f"sigma_I={bits_I}  sigma_J={bits_J}")
            if method == "diffcorr":
                F_s, P_s, M_val = _solve_signed_diffcorr(
                    pb_s, sb_s, list(bits_I), list(bits_J),
                    nF, nP, psi_I_fn, psi_J_fn,
                    coef_bound=coef_bound, base_samples=base_samples,
                    refine_samples=refine_samples,
                    max_iter=30, tol=bisection_tol, verbose=verbose)
            else:
                F_s, P_s, M_val = _solve_signed_bisection(
                    pb_s, sb_s, list(bits_I), list(bits_J),
                    nF, nP, psi_I_fn, psi_J_fn,
                    M_lo=None, M_hi=None,
                    coef_bound=coef_bound, base_samples=base_samples,
                    refine_samples=refine_samples,
                    bisection_tol=bisection_tol, verbose=verbose)
            if F_s is None:
                continue
            if verbose:
                print(f"  -> M={M_val:.5f}")
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
        # best-effort slack estimate
        vals = []
        for (a, b), sig in zip(stopbands, sigma_J):
            ws = np.linspace(a, b, n_dense)
            D = _poly_eval(F, ws) / _poly_eval(P, ws)
            psi = np.array([psi_J_fn(w) for w in ws])
            vals.append(np.abs(D) - psi)
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
        psi = np.array([psi_J_fn(w) for w in ws])
        target = (M + psi)
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
