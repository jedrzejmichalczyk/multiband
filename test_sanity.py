"""
Sanity check: reproduce the Section III toy example of Lunot et al.

  One stopband  J = [1.1, 2]   (truncation of the original [1.1, 2])
  One passband  I = [-1, 1]
  Degree 2 all-pole (nF = 2, nP = 0).

Exact answer is the scaled Chebyshev polynomial T_2(x) = 2 x^2 - 1.
The LP formulation should recover F(x) = x^2 - 1/2 (monic)
and P(x) = const = 1/2 up to normalisation, so F/P = 2 x^2 - 1.
"""
import numpy as np
from multiband_synthesis import solve_zolotarev, poly_eval

sol = solve_zolotarev(
    passbands=[(-1.0, 1.0)],
    stopbands=[(1.1, 2.0)],
    nF=2, nP=0,
    psi_I=1.0, psi_J=0.0,
    base_samples=40, verbose=True,
)

F, P, M = sol["F"], sol["P"], sol["M"]
print(f"F (monic) = {F}")
print(f"P         = {P}")
print(f"min |F/P| on J = {M:.6f}")
print(f"Ratio F/P evaluated -> normalised against leading coeff of F=1:")
print(f"   => equivalent polynomial q(x) = F(x)/P[0] = "
      f"{F / P[0]}")
print(f"   Chebyshev T_2 expected:  [ -1, 0, 2 ]")

# At x = 2 the Chebyshev polynomial gives T_2(2) = 7.
print(f"\nSanity check values:")
for x in (-1.0, 0.0, 1.0, 1.1, 2.0):
    q = poly_eval(F, x) / P[0]
    t2 = 2 * x * x - 1
    print(f"   x={x:+.2f}   F/P·P[0]={q:+.4f}   T_2={t2:+.4f}")
