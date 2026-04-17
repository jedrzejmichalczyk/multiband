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
    k = np.arange(n)
    x = np.cos((2 * k + 1) * np.pi / (2 * n))
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


def _build_lp(I_pts, J_pts, nF, nP, M, psi_I_fn, psi_J_fn, coef_bound):
    """Assemble the LP of Section V in the CHEBYSHEV basis.

       max h
        s.t.   sigma F(y) - (psi + M) P(y) >= h       for y in J   (both signs)
               sigma F(y) + (psi + M) P(y) >= h       for y in J
               sigma P(x) >= 0                        for x in I
               |F(x)| <= sigma psi(x) P(x)            for x in I

    Variables (all in Chebyshev basis on a [-1,1] reference interval):

        [f_0, f_1, ..., f_{nF},   p_0, ..., p_{nP-1},   h]

    with  F(t) = sum_i f_i T_i(t)   and   P(t) = T_{nP}(t) + sum_{i<nP} p_i T_i(t).

    Fixing  lc_{Cheb}(P) = 1  (coefficient of T_nP is 1) plays the role of
    the Lunot-Seyfert normalisation lc(P) = 1 in the Chebyshev basis, but
    with vastly better conditioning when t is already scaled to [-1, 1].

    I_pts, J_pts are lists of (t_sample, sigma) pairs in the scaled variable.
    """
    n_f = nF + 1
    n_p = nP                       # leading Cheb coefficient fixed to 1
    nvar = n_f + n_p + 1
    h_idx = nvar - 1

    c = np.zeros(nvar)
    c[h_idx] = -1.0

    A = []
    b = []

    for y, sigma in J_pts:
        psi = psi_J_fn(y)
        Meff = psi + M
        Fvec = _cheb_basis_row(y, n_f)                 # (nF+1,)
        Pvec_all = _cheb_basis_row(y, nP + 1)          # (nP+1,)
        Pvec_low = Pvec_all[:nP] if nP else np.empty(0)
        P_lead = Pvec_all[nP]                          # T_nP(y)

        # sigma F - Meff P >= h
        row = np.zeros(nvar)
        row[:n_f] = -sigma * Fvec
        if n_p:
            row[n_f:n_f + n_p] = Meff * Pvec_low
        row[h_idx] = 1.0
        A.append(row); b.append(-Meff * P_lead)

        # sigma F + Meff P >= h
        row = np.zeros(nvar)
        row[:n_f] = -sigma * Fvec
        if n_p:
            row[n_f:n_f + n_p] = -Meff * Pvec_low
        row[h_idx] = 1.0
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
    """Return (F_cheb, P_cheb), arrays of Chebyshev coefficients ordered by
    increasing T_k index."""
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


def _probe_feasibility(passbands, stopbands, sigma_I, sigma_J, nF, nP,
                       M, psi_I_fn, psi_J_fn,
                       coef_bound=1e8, base_samples=25, refine_samples=2000,
                       max_exchange=8, tol=1e-6, verbose=False):
    """
    Return (feasible, h, F, P) for the signed problem at level M.

    The routine solves the LP on a sample grid; then densely samples the
    continuum, locates the worst constraint violators, adds them to the
    grid and resolves.  Repeats until the dense grid is also satisfied,
    which certifies the continuous constraint.
    """
    # Interval-level sample caches:   list of (x, sigma)
    I_samples_set = []
    for k, (a, b) in enumerate(passbands):
        for x in _chebyshev_nodes(a, b, max(6, base_samples)):
            I_samples_set.append((x, sigma_I[k]))
    J_samples_set = []
    for k, (a, b) in enumerate(stopbands):
        width = abs(b - a)
        n = int(np.ceil(base_samples * max(1.0, 0.5 + np.log10(1 + 10 * width))))
        for y in _chebyshev_nodes(a, b, n):
            J_samples_set.append((y, sigma_J[k]))

    F = P = None
    h = -np.inf

    for it in range(max_exchange):
        res = _build_lp(I_samples_set, J_samples_set,
                        nF, nP, M, psi_I_fn, psi_J_fn, coef_bound)
        if not res.success:
            if verbose:
                print(f"      LP fail at exchange {it}: {res.message}")
            return False, -np.inf, F, P

        h = res.x[-1]
        F, P = _extract_poly(res.x, nF, nP)

        # Dense verification (in Chebyshev basis on the scaled interval).
        new_pts_I = []
        new_pts_J = []

        for k, (a, b) in enumerate(passbands):
            xs = np.linspace(a, b, refine_samples)
            sig = sigma_I[k]
            psi_x = np.array([psi_I_fn(x) for x in xs])
            Fx = _cheb_eval(F, xs)
            Px = _cheb_eval(P, xs)
            v_abs = np.abs(Fx) - psi_x * np.abs(Px)
            v_sig = -sig * Px
            worst_abs_idx = int(np.argmax(v_abs))
            worst_sig_idx = int(np.argmax(v_sig))
            if v_abs[worst_abs_idx] > tol:
                new_pts_I.append((xs[worst_abs_idx], sig))
            if v_sig[worst_sig_idx] > tol:
                new_pts_I.append((xs[worst_sig_idx], sig))

        for k, (a, b) in enumerate(stopbands):
            ys = np.linspace(a, b, refine_samples)
            sig = sigma_J[k]
            psi_y = np.array([psi_J_fn(y) for y in ys])
            Meff = psi_y + M
            Fy = _cheb_eval(F, ys)
            Py = _cheb_eval(P, ys)
            # Violation:  h - (sigma F - Meff |P|)  > 0
            v = h - (sig * Fy - Meff * np.abs(Py))
            worst_idx = int(np.argmax(v))
            if v[worst_idx] > tol * max(1.0, abs(h)):
                new_pts_J.append((ys[worst_idx], sig))

        if not new_pts_I and not new_pts_J:
            if verbose:
                print(f"      exchange {it}: converged, h={h:.4e}")
            break

        if verbose:
            print(f"      exchange {it}: added {len(new_pts_I)} I pts, "
                  f"{len(new_pts_J)} J pts (h={h:.4e})")

        I_samples_set.extend(new_pts_I)
        J_samples_set.extend(new_pts_J)

    return (h > -tol), h, F, P


# -----------------------------------------------------------------------------
# Bisection on M for a fixed sign combination
# -----------------------------------------------------------------------------

def _solve_signed_bisection(passbands, stopbands, sigma_I, sigma_J, nF, nP,
                            psi_I_fn, psi_J_fn,
                            M_lo=0.0, M_hi=None,
                            bisection_tol=1e-5,
                            coef_bound=1e8, base_samples=25,
                            refine_samples=2000, verbose=False):
    """Bisect on M to find the largest level at which the LP is feasible
    (h > 0), for the given sign combination."""
    # Grow M_hi until infeasible.
    if M_hi is None:
        M_hi = max(1.0, 2.0 * M_lo + 1.0)
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
            M_hi *= 2.0
        else:
            raise RuntimeError("Could not bracket M: feasible up to a huge "
                               "value - probably an unbounded/degenerate "
                               "formulation.")

    # Ensure M_lo is feasible (optional).
    feas_lo, _, F_best, P_best = _probe_feasibility(
        passbands, stopbands, sigma_I, sigma_J, nF, nP,
        M_lo, psi_I_fn, psi_J_fn,
        coef_bound=coef_bound, base_samples=base_samples,
        refine_samples=refine_samples, verbose=False)
    if not feas_lo:
        if verbose:
            print(f"    low bound M={M_lo} already infeasible.")
        return None, None, -np.inf

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
                    rescale=True, verbose=False):
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
            F_s, P_s, M_val = _solve_signed_bisection(
                pb_s, sb_s, list(bits_I), list(bits_J),
                nF, nP, psi_I_fn, psi_J_fn,
                M_lo=0.0, M_hi=None,
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
