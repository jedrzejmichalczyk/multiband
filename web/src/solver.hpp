// Multiband Zolotarev filter synthesis — C++ port of multiband_synthesis.py.
//
// Mirrors the Python reference implementation:
//   - Chebyshev basis on a rescaled variable t = omega / scale
//   - sign enumeration on passband (P) and stopband (F) intervals
//   - per-sign iterative differential correction (eq. 14 of Lunot et al.)
//   - Remez-style exchange refinement of the sample set
//
// The LP back-end is HiGHS (same solver scipy.optimize.linprog uses), so
// numerical answers should match the Python version bit-for-bit at the
// convergence tolerance.
//
// Public entry point:  solve_zolotarev(spec) -> Result.
// For WebAssembly we expose solve_json(const std::string& spec_json) that
// marshals this API over strings.
#pragma once

#include <cmath>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace zolo {

struct Interval { double a; double b; };

using PsiFn = std::function<double(double)>;

struct Spec {
  std::vector<Interval> passbands;   // I = union of passbands
  std::vector<Interval> stopbands;   // J = union of stopbands
  int nF = 0;                        // reflection-zero count (deg F)
  int nP = 0;                        // transmission-zero count (deg P)
  PsiFn psi_I;                       // passband |F/P| <= psi_I(w)
  PsiFn psi_J;                       // stopband baseline: |F/P| >= M + psi_J(w)
  bool rescale = true;               // map intervals into [-1, 1]
  int base_samples = 40;             // initial Chebyshev samples per interval
  int refine_samples = 3000;         // dense grid for Remez verification
  int max_iter = 30;                 // diff-correction iteration cap
  int max_exchange = 12;             // Remez exchange cap per LP
  double coef_bound = 1e6;           // box bound on Chebyshev coefficients
  double tol = 1e-5;                 // convergence / feasibility tolerance
};

// Polynomial coefficients are returned in MONOMIAL basis, ascending order:
//   F(omega) = sum_i F_mono[i] * omega^i
struct Result {
  bool success = false;
  std::string message;
  std::vector<double> F_mono;
  std::vector<double> P_mono;
  double M = -INFINITY;              // best slack above psi_J achieved
  std::vector<int> sigma_I;
  std::vector<int> sigma_J;
  double scale = 1.0;
};

// --- Main solver ------------------------------------------------------------

Result solve_zolotarev(const Spec& spec);

// --- Conveniences for the web UI --------------------------------------------

// Evaluate filter response at a set of omega points.  Returns |S21|^2 and
// |S11|^2 in output vectors (caller resizes).
struct Response {
  std::vector<double> omega;
  std::vector<double> S21_sq;
  std::vector<double> S11_sq;
};
Response scattering_response(const std::vector<double>& F_mono,
                             const std::vector<double>& P_mono,
                             const std::vector<double>& omega);

// Convert return loss / rejection from dB to the corresponding psi value
// (max allowed or minimum required |F/P|).
double rl_db_to_psi(double rl_db);
double rej_db_to_psi(double rej_db);

// Toggle verbose iteration tracing on stderr (off by default).
void set_trace(bool v);

// JSON entry point for the WebAssembly build.
// Input (example):
//   {"passbands": [[-1,-0.625],[0.25,1]],
//    "stopbands": [[-10,-1.188],[-0.5,0.125],[1.212,10]],
//    "nF": 9, "nP": 3,
//    "psi_I_db": 20,
//    "psi_J_pieces": [[[-0.5,0.125],30]],  // overrides in intervals
//    "psi_J_default_db": 15,
//    "evaluation_omega": [-2.0,-1.9,...,2.0]}
// Output:
//   {"success": true/false, "message": "...",
//    "F_mono": [...], "P_mono": [...], "M": ...,
//    "sigma_I": [...], "sigma_J": [...],
//    "scale": 10.0,
//    "response": {"omega":[...], "S21_dB":[...], "S11_dB":[...]}}
std::string solve_json(const std::string& spec_json);

}  // namespace zolo
