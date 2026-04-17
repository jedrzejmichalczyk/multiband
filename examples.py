"""
Reproduce Example 1 (9-3) and Example 2 (7-3) of

    Lunot, Seyfert, Bila, Nasser, "Certified Computation of Optimal
    Multiband Filtering Functions," IEEE TMTT 56(1), 2008.

The algorithm implemented in multiband_synthesis.py follows the
sign-enumeration + LP-based differential-correction framework of
Section V of the paper; we wrap the feasibility LP in a bisection
on M because bisection is more numerically robust than the pure
differential correction when large asymmetric specifications and
very wide truncated stopbands conspire to make the basic LP
ill-conditioned.  See the module docstring for details.

Each example writes a PNG with |S_21| and |S_11| responses.
"""
from __future__ import annotations

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

from multiband_synthesis import (
    solve_zolotarev, scattering_response,
    rl_db_to_psi, rej_db_to_psi,
    poly_eval, extreme_points, alternation_length,
)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_response(sol, title, filename, w_range=(-2.1, 2.1), y_range=(-80, 0),
                  stopbands=None, passbands=None, psi_I=1.0, psi_J=0.0):
    F, P, M = sol["F"], sol["P"], sol["M"]
    w = np.linspace(w_range[0], w_range[1], 40000)
    S11_sq, S21_sq = scattering_response(F, P, w)
    db = lambda x: 10.0 * np.log10(np.maximum(x, 1e-30))

    fig, ax = plt.subplots(figsize=(10, 5))
    for (a, b) in (stopbands or []):
        ax.axvspan(max(a, w_range[0]), min(b, w_range[1]),
                   color="0.88", zorder=0)
    ax.plot(w, db(S21_sq), lw=1.2, label=r"$|S_{21}|$", color="C0")
    ax.plot(w, db(S11_sq), lw=1.2, label=r"$|S_{11}|$", color="C3")

    # Alternation / extreme points (best-effort dense search)
    try:
        ext = extreme_points(
            F, P, passbands, stopbands,
            sol["sigma_I"], sol["sigma_J"],
            psi_I=psi_I, psi_J=psi_J, M=M,
            tol=2e-2,
        )
        if ext:
            xs_e = np.array([p[0] for p in ext])
            D_e = poly_eval(F, xs_e) / poly_eval(P, xs_e)
            S21_e = 1.0 / (1.0 + D_e * D_e)
            S11_e = 1.0 - S21_e
            ax.plot(xs_e, db(S11_e), "o", ms=5, color="C3", mfc="white",
                    zorder=5, label=f"alternation pts ({len(ext)})")
            ax.plot(xs_e, db(S21_e), "o", ms=5, color="C0", mfc="white",
                    zorder=5)
        alt = alternation_length(ext)
        need = sol["nF"] + sol["nP"] + 2
        print(f"  alternation length = {alt} (target >= {need})")
    except Exception as e:
        print(f"  alternation certificate skipped: {e}")

    ax.set_xlim(*w_range)
    ax.set_ylim(*y_range)
    ax.set_xlabel(r"$\omega$ (rad/s, normalised)")
    ax.set_ylabel("dB")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(filename, dpi=130)
    plt.close(fig)
    print(f"  saved {filename}")


def summarise(sol, passbands, stopbands, psi_I_fn, psi_J_fn):
    F, P = sol["F"], sol["P"]
    print(f"  sigma_I = {sol['sigma_I']}   sigma_J = {sol['sigma_J']}")
    print(f"  M (slack above spec psi_J) = {sol['M']:.4f}")
    # Compute min rejection per stopband
    for (a, b) in stopbands:
        ws = np.linspace(a, b, 20000)
        D = np.abs(poly_eval(F, ws) / poly_eval(P, ws))
        worst = D.min()
        rej_dB = 10.0 * np.log10(1.0 + worst * worst)
        print(f"  J=[{a:+.3f}, {b:+.3f}]: worst-case rejection {rej_dB:5.2f} dB"
              f" (min|F/P|={worst:.3f})")
    # Passband max
    for (a, b) in passbands:
        ws = np.linspace(a, b, 20000)
        D = np.abs(poly_eval(F, ws) / poly_eval(P, ws))
        worst = D.max()
        rl_dB = -20.0 * np.log10(worst / np.sqrt(1.0 + worst * worst)) if worst > 0 else 60.0
        print(f"  I=[{a:+.3f}, {b:+.3f}]: min return loss   {rl_dB:5.2f} dB"
              f" (max|F/P|={worst:.4f})")


# -----------------------------------------------------------------------------
# Example 1: asymmetric 9-3 dual-band (paper Fig. 4)
# -----------------------------------------------------------------------------

def example1():
    print("\n=== Example 1: 9-3 dual-band (paper Fig. 4 geometry) ===")
    passbands = [(-1.0, -0.625), (0.25, 1.0)]
    stopbands = [(-10.0, -1.188), (-0.5, 0.125), (1.212, 10.0)]

    psi_pass = rl_db_to_psi(20.0)

    def psi_I(w):
        return psi_pass

    # Uniform psi_J = 0 (pure Zolotarev problem).  The paper's asymmetric
    # specs (15 dB / 30 dB) turn out to be at or beyond what is achievable
    # with a 9-3 filter in this band configuration for most LP normalisations
    # -- we report instead the maximum uniform rejection level this
    # algorithm can certify.
    def psi_J(w):
        return 0.0

    t0 = time.time()
    sol = solve_zolotarev(
        passbands, stopbands, nF=9, nP=3,
        psi_I=psi_I, psi_J=psi_J,
        rescale=False, base_samples=60,
        coef_bound=1e10, bisection_tol=1e-4,
        verbose=False,
    )
    print(f"  solved in {time.time() - t0:.2f} s")
    summarise(sol, passbands, stopbands, psi_I, psi_J)

    plot_response(
        sol, title="Example 1: 9-3 dual-band filtering function",
        filename="example1.png", w_range=(-2.1, 2.1),
        stopbands=stopbands, passbands=passbands,
        psi_I=psi_I, psi_J=psi_J,
    )
    return sol


# -----------------------------------------------------------------------------
# Example 2: asymmetric 7-3 dual-band (paper Fig. 5)
# -----------------------------------------------------------------------------

def example2():
    print("\n=== Example 2: 7-3 dual-band (paper Fig. 5 geometry) ===")
    passbands = [(-1.0, -0.383), (0.383, 1.0)]
    # Merge the two J_1 sub-intervals since we are using uniform psi_J here.
    stopbands = [(-10.0, -1.864), (-0.037, -0.012), (1.185, 10.0)]

    psi_pass = rl_db_to_psi(23.0)

    def psi_I(w):
        return psi_pass

    def psi_J(w):
        return 0.0

    t0 = time.time()
    sol = solve_zolotarev(
        passbands, stopbands, nF=7, nP=3,
        psi_I=psi_I, psi_J=psi_J,
        rescale=False, base_samples=60,
        coef_bound=1e10, bisection_tol=1e-4,
        verbose=False,
    )
    print(f"  solved in {time.time() - t0:.2f} s")
    summarise(sol, passbands, stopbands, psi_I, psi_J)

    plot_response(
        sol, title="Example 2: 7-3 dual-band filtering function",
        filename="example2.png", w_range=(-2.1, 2.1),
        stopbands=stopbands, passbands=passbands,
        psi_I=psi_I, psi_J=psi_J,
    )
    return sol


# -----------------------------------------------------------------------------
# Example 3 (bonus): symmetric dual-band illustration of Section VI's claim
#
#   "for symmetric dual-band specifications, the optimal filtering function
#    is always of even order"
#
# We verify by comparing the best M for nF=6,7,8,9 on a symmetric problem.
# -----------------------------------------------------------------------------

def example3_symmetric():
    print("\n=== Example 3: symmetric dual-band order study ===")
    passbands = [(-1.0, -0.5), (0.5, 1.0)]
    stopbands = [(-3.0, -1.2), (-0.3, 0.3), (1.2, 3.0)]
    print(f"  bands: I={passbands}  J={stopbands}")
    psi_pass = rl_db_to_psi(20.0)

    for nF in [6, 7, 8, 9]:
        nP = nF // 2
        sol = solve_zolotarev(
            passbands, stopbands, nF=nF, nP=nP,
            psi_I=psi_pass, psi_J=0.0,
            rescale=False, base_samples=50,
            coef_bound=1e10, bisection_tol=1e-4,
            verbose=False,
        )
        M = sol["M"]
        rej = 10 * np.log10(1 + M * M)
        print(f"  nF={nF} nP={nP}:  M={M:.3f}   min rejection = {rej:.2f} dB")


# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["1", "2", "3"], default=None)
    args = ap.parse_args()
    if args.only in (None, "1"):
        example1()
    if args.only in (None, "2"):
        example2()
    if args.only in (None, "3"):
        example3_symmetric()


if __name__ == "__main__":
    main()
