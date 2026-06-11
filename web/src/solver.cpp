// Multiband Zolotarev filter synthesis — port of multiband_synthesis.py.
// See solver.hpp for the API contract.
#include "solver.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <sstream>
#include <tuple>

#include "Highs.h"

// Verbose iteration tracing switch (set via `zolo::set_trace(true)` from
// the JSON layer).  Used to compare against the Python reference.
#include <cstdlib>
#include <cstdio>
namespace {
  bool g_trace = false;
}
int zolo_trace() { return g_trace ? 1 : 0; }
namespace zolo { void set_trace(bool v); }
void zolo::set_trace(bool v) { g_trace = v; }

namespace zolo {

// ---------------------------------------------------------------------------
// Chebyshev basis helpers
// ---------------------------------------------------------------------------

// Fill out[0..n-1] with T_0(x), ..., T_{n-1}(x).
static void cheb_basis(double x, int n, std::vector<double>& out) {
  out.assign(n, 0.0);
  if (n >= 1) out[0] = 1.0;
  if (n >= 2) out[1] = x;
  for (int k = 2; k < n; ++k) out[k] = 2.0 * x * out[k - 1] - out[k - 2];
}

// Evaluate  sum_i c[i] * T_i(x)  at x.
static double cheb_eval(const std::vector<double>& c, double x) {
  // Clenshaw
  double b2 = 0.0, b1 = 0.0;
  for (int k = static_cast<int>(c.size()) - 1; k >= 1; --k) {
    double tmp = 2.0 * x * b1 - b2 + c[k];
    b2 = b1;
    b1 = tmp;
  }
  return x * b1 - b2 + c[0];
}

// Convert Chebyshev coefficients to monomial coefficients (ascending).
// Uses the standard identity T_n(x) = cos(n acos x) and cheb2poly relation.
static std::vector<double> cheb_to_mono(const std::vector<double>& c) {
  int n = static_cast<int>(c.size());
  if (n == 0) return {};
  if (n == 1) return {c[0]};
  // We evaluate the change of basis using the recurrence
  //   T_k(x) = 2 x T_{k-1}(x) - T_{k-2}(x).
  // Maintain the two last polynomials in monomial basis.
  std::vector<double> prev(n, 0.0), curr(n, 0.0);
  prev[0] = 1.0;       // T_0 = 1
  curr[1] = 1.0;       // T_1 = x
  std::vector<double> out(n, 0.0);
  out[0] = c[0] * prev[0];
  if (n >= 2) out[1] = c[1] * curr[1];
  std::vector<double> next(n, 0.0);
  for (int k = 2; k < n; ++k) {
    std::fill(next.begin(), next.end(), 0.0);
    // next = 2*x*curr - prev
    for (int i = 0; i < n - 1; ++i) next[i + 1] += 2.0 * curr[i];
    for (int i = 0; i < n; ++i) next[i] -= prev[i];
    for (int i = 0; i < n; ++i) out[i] += c[k] * next[i];
    std::swap(prev, curr);
    std::swap(curr, next);
  }
  return out;
}

// Rescale monomial coefficients  sum c_i t^i  with t = omega / scale
// back into coefficients of omega.
static std::vector<double> rescale_mono(const std::vector<double>& c,
                                        double inv_scale) {
  std::vector<double> out(c.size(), 0.0);
  double f = 1.0;
  for (std::size_t i = 0; i < c.size(); ++i) {
    out[i] = c[i] * f;
    f *= inv_scale;
  }
  return out;
}

// Evaluate a monomial polynomial (ascending).
static double polyval(const std::vector<double>& c, double x) {
  double acc = 0.0;
  for (std::size_t i = c.size(); i-- > 0;) acc = acc * x + c[i];
  return acc;
}

// Chebyshev-Lobatto nodes (INCLUDE the endpoints).  First-kind nodes
// leave a gap near both boundaries which the LP can exploit by placing a
// reflection zero exactly on a truncation edge, giving spurious "optimal"
// filters -- Lobatto closes that loophole.
static std::vector<double> chebyshev_nodes(double a, double b, int n) {
  if (n < 2) return {0.5 * (a + b)};
  std::vector<double> out(n);
  for (int k = 0; k < n; ++k) {
    double c = std::cos(M_PI * k / (n - 1));
    out[k] = 0.5 * (a + b) + 0.5 * (b - a) * c;
  }
  return out;
}

// Multiplicative stopband weight of the paper's Section V.D spec
// mechanism: the rejection target psi_J scales the criterion, with a
// weight of 1 where no target is given (psi_J = 0) so that the classical
// unweighted Zolotarev problem is recovered there.
static double stopband_weight(double psi) {
  return psi > 1e-9 ? psi : 1.0;
}

// ---------------------------------------------------------------------------
// LP assembly (HiGHS)
// ---------------------------------------------------------------------------

struct SamplePt {
  double x;
  int sigma;
};

// Returns (h, F_cheb, P_cheb) or (nan, {}, {}) on LP failure.
// The LP is in the Chebyshev basis with P's leading Cheb coefficient pinned
// to 1 (classical lc(P)=1 normalization).
struct LPResult {
  bool success;
  double h;
  std::vector<double> F;   // F Cheb coeffs, length nF+1
  std::vector<double> P;   // P Cheb coeffs, length nP+1 (leading = 1)
};

// One LP at a fixed level M that GROWS across Remez exchange iterations.
// The HiGHS instance persists so each exchange iteration only appends the
// violator rows -- but every solve runs from scratch (clearSolver):
// warm-starting from the previous iteration's basis was measured to
// produce false "infeasible" verdicts on this ill-conditioned model,
// which stop DC and mint bogus optimality certificates.
class LpSession {
 public:
  LpSession(int nF, int nP, double M,
            const PsiFn& psi_I, const PsiFn& psi_J,
            double coef_bound, double tol,
            const std::vector<double>* F_ref)
      : n_f_(nF + 1), n_p_(nP), nP_(nP), M_(M), tol_(tol),
        psi_I_(psi_I), psi_J_(psi_J), F_ref_(F_ref) {
    nvar_ = n_f_ + n_p_ + 1;
    h_idx_ = nvar_ - 1;
    highs_.setOptionValue("output_flag", false);
    for (int j = 0; j < nvar_; ++j) {
      const bool is_h = (j == h_idx_);
      // F, P coefficients in [-coef_bound, coef_bound]; maximise h =>
      // minimise -h.  h is capped from above: at strongly feasible
      // levels (all Meff clamped to 0) the LP would otherwise ride F to
      // the coefficient box, which HiGHS reports as unbounded.  Only
      // h > tol matters to callers, so the cap never changes a verdict.
      HighsStatus rc = highs_.addCol(is_h ? -1.0 : 0.0,
                                     is_h ? -kHighsInf : -coef_bound,
                                     is_h ? 1e6 : +coef_bound,
                                     0, nullptr, nullptr);
      if (rc != HighsStatus::kOk && rc != HighsStatus::kWarning)
        broken_ = true;
    }
    highs_.changeObjectiveSense(ObjSense::kMinimize);
  }

  // Stopband sample: two rows linearising  sig F - Meff |P| >= h*rhs
  // with the multiplicative weight Meff = M * w_J (paper Section V.D).
  // The pair encodes that constraint ONLY for Meff >= 0; with Meff < 0
  // their intersection would demand sig F >= h + |Meff||P| -- a
  // stricter, wrong constraint that silently corrupts every probe at
  // negative levels.  The true constraint is vacuous there, so clamp:
  // the rows then just pin the assumed sign, sig F >= h*rhs.
  void add_J(const SamplePt& s) {
    double y = s.x;
    int sig = s.sigma;
    double Meff = std::max(stopband_weight(psi_J_(y)) * M_, 0.0);
    cheb_basis(y, n_f_, Fvec_);
    cheb_basis(y, nP_ + 1, Pvec_);
    double P_lead = Pvec_[nP_];
    double rhs = 1.0;
    if (F_ref_ != nullptr) {
      rhs = std::max(std::abs(cheb_eval(*F_ref_, y)), 1e-12);
    }
    for (int side = -1; side <= 1; side += 2) {
      // -sig Fvec.f  -+  Meff Pvec_low.p  +  rhs*h  <=  +-Meff P_lead
      idx_.clear(); val_.clear();
      for (int i = 0; i < n_f_; ++i) {
        idx_.push_back(i);
        val_.push_back(-sig * Fvec_[i]);
      }
      for (int i = 0; i < n_p_; ++i) {
        idx_.push_back(n_f_ + i);
        val_.push_back(-side * Meff * Pvec_[i]);
      }
      idx_.push_back(h_idx_);
      val_.push_back(rhs);
      add_row(side * Meff * P_lead);
    }
  }

  // Passband sample: sigma P >= 0,  F <= sig*psi*P,  -F <= sig*psi*P.
  void add_I(const SamplePt& s) {
    double x = s.x;
    int sig = s.sigma;
    double psi = psi_I_(x);
    cheb_basis(x, n_f_, Fvec_);
    cheb_basis(x, nP_ + 1, Pvec_);
    double P_lead = Pvec_[nP_];

    // sig*P >= 0  =>  -sig Pvec_low.p  <=  sig*P_lead
    idx_.clear(); val_.clear();
    for (int i = 0; i < n_p_; ++i) {
      idx_.push_back(n_f_ + i);
      val_.push_back(-sig * Pvec_[i]);
    }
    add_row(sig * P_lead);

    // +-F - sig*psi*P <= 0  =>  +-Fvec.f - sig*psi Pvec_low.p <= sig*psi P_lead
    for (int side = 1; side >= -1; side -= 2) {
      idx_.clear(); val_.clear();
      for (int i = 0; i < n_f_; ++i) {
        idx_.push_back(i);
        val_.push_back(side * Fvec_[i]);
      }
      for (int i = 0; i < n_p_; ++i) {
        idx_.push_back(n_f_ + i);
        val_.push_back(-sig * psi * Pvec_[i]);
      }
      add_row(sig * psi * P_lead);
    }
  }

  LPResult solve() {
    if (broken_) {
      if (zolo_trace()) std::fprintf(stderr, "  LP session broken\n");
      return {false, 0, {}, {}};
    }
    highs_.clearSolver();
    return run_once();
  }

 private:
  LPResult run_once() {
    HighsStatus st = highs_.run();
    if (st != HighsStatus::kOk && st != HighsStatus::kWarning) {
      if (zolo_trace())
        std::fprintf(stderr, "  LP run status = %d\n", static_cast<int>(st));
      return {false, 0, {}, {}};
    }
    HighsModelStatus ms = highs_.getModelStatus();
    if (ms != HighsModelStatus::kOptimal) {
      if (zolo_trace())
        std::fprintf(stderr, "  LP model status = %d\n",
                     static_cast<int>(ms));
      return {false, 0, {}, {}};
    }
    const auto& sol = highs_.getSolution();
    LPResult out;
    out.success = true;
    out.h = sol.col_value[h_idx_];
    out.F.assign(sol.col_value.begin(), sol.col_value.begin() + n_f_);
    out.P.assign(n_p_ + 1, 0.0);
    for (int i = 0; i < n_p_; ++i) out.P[i] = sol.col_value[n_f_ + i];
    out.P[n_p_] = 1.0;
    return out;
  }

  void add_row(double ub) {
    HighsStatus rc = highs_.addRow(-kHighsInf, ub,
                                   static_cast<HighsInt>(idx_.size()),
                                   idx_.data(), val_.data());
    // kWarning is fine: HiGHS warns while filtering explicit zeros
    // (e.g. the P-block of a clamped Meff = 0 stopband row).
    if (rc != HighsStatus::kOk && rc != HighsStatus::kWarning)
      broken_ = true;
  }

  Highs highs_;
  int n_f_, n_p_, nP_, nvar_, h_idx_;
  double M_;
  double tol_;
  const PsiFn& psi_I_;
  const PsiFn& psi_J_;
  const std::vector<double>* F_ref_;
  bool broken_ = false;
  // scratch
  std::vector<double> Fvec_, Pvec_, val_;
  std::vector<HighsInt> idx_;
};

// ---------------------------------------------------------------------------
// Remez-style exchange loop (one LP call with adaptive sampling)
// ---------------------------------------------------------------------------

static std::tuple<double, std::vector<double>, std::vector<double>>
probe_lp(const Spec& spec,
         const std::vector<Interval>& passbands,
         const std::vector<Interval>& stopbands,
         const std::vector<int>& sigma_I,
         const std::vector<int>& sigma_J,
         double M,
         const std::vector<double>* F_ref,
         const PsiFn& psi_I, const PsiFn& psi_J,
         int max_exchange, int base_samples) {
  // Fixed-count sampling per interval (no width scaling).  Width scaling
  // over-constrains wide outer stopbands relative to narrow middle
  // stopbands and biases the LP's rejection allocation.
  const int n_pts = std::max(6, base_samples);
  LpSession lp(spec.nF, spec.nP, M, psi_I, psi_J, spec.coef_bound,
               spec.tol, F_ref);
  for (size_t k = 0; k < passbands.size(); ++k) {
    auto xs = chebyshev_nodes(passbands[k].a, passbands[k].b, n_pts);
    for (double x : xs) lp.add_I({x, sigma_I[k]});
  }
  for (size_t k = 0; k < stopbands.size(); ++k) {
    auto ys = chebyshev_nodes(stopbands[k].a, stopbands[k].b, n_pts);
    for (double y : ys) lp.add_J({y, sigma_J[k]});
  }

  double h = -std::numeric_limits<double>::infinity();
  std::vector<double> F, P;

  for (int it = 0; it < max_exchange; ++it) {
    auto res = lp.solve();
    if (!res.success) return {h, F, P};
    h = res.h;
    F = std::move(res.F);
    P = std::move(res.P);

    // Dense verification; violators become new rows of the same LP
    // (warm-started re-solve).
    bool added = false;

    for (size_t k = 0; k < passbands.size(); ++k) {
      int n = spec.refine_samples;
      double a = passbands[k].a, b = passbands[k].b;
      int sig = sigma_I[k];
      double best_ratio = -std::numeric_limits<double>::infinity();
      double best_ratio_x = 0;
      double best_sig = -std::numeric_limits<double>::infinity();
      double best_sig_x = 0;
      for (int i = 0; i < n; ++i) {
        double x = a + (b - a) * i / (n - 1);
        double Fx = cheb_eval(F, x);
        double Px = cheb_eval(P, x);
        double psi = psi_I(x);
        // Ratio-relative violation: |P| is tiny on I relative to the
        // huge F values on J, so an absolute threshold lets the LP
        // cheat the return-loss constraint by percents between samples.
        double vr = (std::abs(Fx) - psi * std::abs(Px))
                        / std::max(std::abs(Px), 1e-12)
                    - 10.0 * spec.tol * std::max(1.0, psi);
        double vs = -sig * Px;
        if (vr > best_ratio) { best_ratio = vr; best_ratio_x = x; }
        if (vs > best_sig)   { best_sig = vs;   best_sig_x = x; }
      }
      if (best_ratio > 0)        { lp.add_I({best_ratio_x, sig}); added = true; }
      if (best_sig   > spec.tol) { lp.add_I({best_sig_x, sig});   added = true; }
    }

    for (size_t k = 0; k < stopbands.size(); ++k) {
      int n = spec.refine_samples;
      double a = stopbands[k].a, b = stopbands[k].b;
      int sig = sigma_J[k];
      double best_v = -std::numeric_limits<double>::infinity();
      double best_v_x = 0;
      for (int i = 0; i < n; ++i) {
        double y = a + (b - a) * i / (n - 1);
        double Fy = cheb_eval(F, y);
        double Py = cheb_eval(P, y);
        double Meff = std::max(stopband_weight(psi_J(y)) * M, 0.0);
        // (clamped exactly like the LP rows)
        double rhs = (F_ref ? std::abs(cheb_eval(*F_ref, y)) : 1.0);
        double v = h * rhs - (sig * Fy - Meff * std::abs(Py));
        if (v > best_v) { best_v = v; best_v_x = y; }
      }
      if (best_v > spec.tol * std::max(1.0, std::abs(h))) {
        lp.add_J({best_v_x, sig});
        added = true;
      }
    }

    if (!added) break;
  }
  return {h, F, P};
}

// probe_lp with a mirror retry.  (sigma_I, sigma_J) and (sigma_I, -sigma_J)
// describe the SAME subproblem under F -> -F (the passband constraints
// |F| <= psi sigma P and sigma P >= 0 are invariant), but the LP is
// ill-conditioned -- polynomial values span many orders of magnitude across
// the bands -- and the two mirrored models take different numerical paths
// through the solver.  One mirror regularly succeeds where the other returns
// a bogus near-zero h or fails outright.  Whenever the first attempt does
// not come back clearly feasible we retry mirrored and keep the better
// outcome (mapping the witness back via F -> -F).  An infeasibility
// certificate therefore requires BOTH mirrors to agree.  |F_ref| is
// mirror-invariant, so the same reference works for both.
static std::tuple<double, std::vector<double>, std::vector<double>>
probe_lp_robust(const Spec& spec,
                const std::vector<Interval>& passbands,
                const std::vector<Interval>& stopbands,
                const std::vector<int>& sigma_I,
                const std::vector<int>& sigma_J,
                double M,
                const std::vector<double>* F_ref,
                const PsiFn& psi_I, const PsiFn& psi_J,
                int max_exchange, int base_samples) {
  auto r1 = probe_lp(spec, passbands, stopbands, sigma_I, sigma_J, M,
                     F_ref, psi_I, psi_J, max_exchange, base_samples);
  if (std::get<0>(r1) > spec.tol) return r1;
  std::vector<int> sJ_m(sigma_J.size());
  for (size_t k = 0; k < sigma_J.size(); ++k) sJ_m[k] = -sigma_J[k];
  auto r2 = probe_lp(spec, passbands, stopbands, sigma_I, sJ_m, M,
                     F_ref, psi_I, psi_J, max_exchange, base_samples);
  if (std::get<0>(r2) > std::get<0>(r1)) {
    for (double& c : std::get<1>(r2)) c = -c;
    return r2;
  }
  return r1;
}

// ---------------------------------------------------------------------------
// Signed optimum via iterative differential correction
// ---------------------------------------------------------------------------

// Dense (continuum-verified) value of the weighted criterion:
//   min over J of  sigma F / (|P| w_J).
static double min_signed_ratio_minus_psi(const std::vector<double>& F,
                                         const std::vector<double>& P,
                                         const std::vector<Interval>& stopbands,
                                         const std::vector<int>& sigma_J,
                                         const PsiFn& psi_J,
                                         int n_samples) {
  double best = std::numeric_limits<double>::infinity();
  for (size_t k = 0; k < stopbands.size(); ++k) {
    double a = stopbands[k].a, b = stopbands[k].b;
    int sig = sigma_J[k];
    for (int i = 0; i < n_samples; ++i) {
      double y = a + (b - a) * i / (n_samples - 1);
      double Fy = cheb_eval(F, y);
      double Py = cheb_eval(P, y);
      if (std::abs(Py) < 1e-14) continue;
      double r = sig * Fy / (std::abs(Py) * stopband_weight(psi_J(y)));
      if (r < best) best = r;
    }
  }
  return best;
}

// Make a candidate honour |F| <= psi sigma P on the WHOLE of I, not just
// at the LP samples: an LP iterate (un-exchanged seed, or a probe whose
// exchange budget ran out) can cheat between passband samples while
// showing a large stopband slack.  Scaling F down by the worst dense
// ratio restores validity at a proportional cost in slack.  Returns the
// factor to multiply F by (1.0 when already valid), or NaN when
// sigma P <= 0 somewhere on I (not repairable by scaling).
static double passband_repair_scale(const std::vector<double>& F,
                                    const std::vector<double>& P,
                                    const std::vector<Interval>& passbands,
                                    const std::vector<int>& sigma_I,
                                    const PsiFn& psi_I,
                                    int n_samples) {
  double ratio_max = 1.0;
  for (size_t k = 0; k < passbands.size(); ++k) {
    double a = passbands[k].a, b = passbands[k].b;
    int sig = sigma_I[k];
    for (int i = 0; i < n_samples; ++i) {
      double x = a + (b - a) * i / (n_samples - 1);
      double sP = sig * cheb_eval(P, x);
      if (sP <= 0.0) return std::numeric_limits<double>::quiet_NaN();
      double r = std::abs(cheb_eval(F, x)) / (psi_I(x) * sP);
      ratio_max = std::max(ratio_max, r);
    }
  }
  if (ratio_max <= 1.0) return 1.0;
  return 1.0 / (ratio_max * (1.0 + 1e-12));
}

struct SignedResult {
  bool success = false;
  double M = -std::numeric_limits<double>::infinity();
  // Certified bound: no candidate with this sign pattern beats M_upper,
  // even on the sampled problem (-inf when the pattern admits none).
  double M_upper = -std::numeric_limits<double>::infinity();
  std::vector<double> F, P;  // Cheb
};

// Eq. (14) quadratic-correction loop from the seed (F0, P0, M0); updates
// `best` with any improvement found.
static void dc_iterate(const Spec& spec,
                       const std::vector<Interval>& pb,
                       const std::vector<Interval>& sb,
                       const std::vector<int>& sI,
                       const std::vector<int>& sJ,
                       const PsiFn& psi_I, const PsiFn& psi_J,
                       std::vector<double> F_km1, std::vector<double> P_km1,
                       double M_km1, SignedResult& best) {
  // Repair the candidate to dense passband validity, re-measure its
  // slack, and keep it when it beats the current best.
  auto consider = [&](const std::vector<double>& F,
                      const std::vector<double>& P) {
    double s = passband_repair_scale(F, P, pb, sI, psi_I,
                                     spec.refine_samples);
    if (std::isnan(s)) return;
    std::vector<double> F_rep = F;
    if (s < 1.0) for (double& c : F_rep) c *= s;
    double M_rep = min_signed_ratio_minus_psi(F_rep, P, sb, sJ, psi_J,
                                              spec.refine_samples);
    if (M_rep > best.M) {
      best.M = M_rep; best.F = std::move(F_rep); best.P = P;
    }
  };
  consider(F_km1, P_km1);
  for (int k = 1; k <= spec.max_iter; ++k) {
    auto [h, F_k, P_k] = probe_lp_robust(spec, pb, sb, sI, sJ, M_km1,
                                         &F_km1, psi_I, psi_J,
                                         spec.max_exchange,
                                         spec.base_samples);
    if (zolo_trace())
      std::fprintf(stderr, "    iter %d: h=%.3e\n", k, h);
    if (F_k.empty() || h <= spec.tol) break;
    double M_k = min_signed_ratio_minus_psi(F_k, P_k, sb, sJ, psi_J,
                                            spec.refine_samples);
    if (zolo_trace())
      std::fprintf(stderr, "            M=%.5f  dM=%.2e\n",
                   M_k, M_k - M_km1);
    consider(F_k, P_k);
    if (M_k - M_km1 < spec.tol * std::max(1.0, std::abs(M_k))) break;
    F_km1 = std::move(F_k);
    P_km1 = std::move(P_k);
    M_km1 = M_k;
  }
}

static SignedResult solve_signed(const Spec& spec,
                                 const std::vector<Interval>& pb,
                                 const std::vector<Interval>& sb,
                                 const std::vector<int>& sI,
                                 const std::vector<int>& sJ,
                                 const PsiFn& psi_I,
                                 const PsiFn& psi_J,
                                 double prune_below) {
  // Step 0: try a sequence of initial M values -- start aggressive
  // (M=1, spec exactly met) and relax until the first LP probe gives
  // h>tol, with an always-feasible negative fallback.  The Remez
  // exchange stays OFF
  // (max_exchange=1) during initialisation: starting from the Lobatto
  // samples the first LP already gives a usable seed, and adding
  // violators at this stage only tightens the problem until it goes
  // infeasible at the very margins we are bracketing (same reasoning as
  // the Python reference).
  // In the multiplicative weighting any level < 0 clamps every stopband
  // row to the vacuous sign-pinning form, so a single negative fallback
  // suffices (it is always feasible whenever the sign pattern admits a
  // candidate at all).
  const double initial_margins[] = {1.0, 0.3, 0.1, 0.03, 0.01, 0.0, -1.0};

  double h0 = -INFINITY, M_start = 0;
  std::vector<double> F_km1, P_km1;
  for (double trial_M : initial_margins) {
    auto [h_try, F_try, P_try] = probe_lp_robust(spec, pb, sb, sI, sJ,
                                                 trial_M, nullptr,
                                                 psi_I, psi_J,
                                                 /*max_exchange=*/1,
                                                 spec.base_samples);
    if (F_try.empty()) continue;
    if (h_try > spec.tol || trial_M < 0) {
      h0 = h_try; F_km1 = F_try; P_km1 = P_try; M_start = trial_M;
      if (h_try > spec.tol) break;
    }
  }
  if (F_km1.empty()) {
    // The M-independent passband constraints admit no candidate for
    // this sign pattern.
    if (zolo_trace()) std::fprintf(stderr, "    init probe failed\n");
    return {};
  }
  double M_km1 = min_signed_ratio_minus_psi(F_km1, P_km1, sb, sJ, psi_J,
                                            spec.refine_samples);
  // The un-exchanged seed cheats between its samples, overstating its
  // slack; starting DC at the overstated level makes the very first
  // probe infeasible and kills the iteration.  Start from the honest
  // (repaired) level instead.
  {
    double s = passband_repair_scale(F_km1, P_km1, pb, sI, psi_I,
                                     spec.refine_samples);
    if (!std::isnan(s) && s < 1.0) {
      std::vector<double> F0_rep = F_km1;
      for (double& c : F0_rep) c *= s;
      M_km1 = min_signed_ratio_minus_psi(F0_rep, P_km1, sb, sJ, psi_J,
                                         spec.refine_samples);
    }
  }
  if (zolo_trace())
    std::fprintf(stderr, "    init M_start=%.3f h0=%.3e M_0=%.5f\n",
                 M_start, h0, M_km1);

  SignedResult best;
  dc_iterate(spec, pb, sb, sI, sJ, psi_I, psi_J,
             F_km1, P_km1, M_km1, best);  // copies: seed reused below

  // ------------------------------------------------------------------
  // Feasibility-kick refinement (global escape + optimality bracket).
  //
  // The DC iteration is fast but only locally reliable: the LP has
  // massively degenerate optimal faces, and which vertex the backend
  // happens to return decides which valley DC follows (observed: the
  // same sign combination converging to M=1.62 with one HiGHS build and
  // M=15.16 with another).  The feasibility test at a fixed level L has
  // no such ambiguity:
  //     h > tol     =>  a strictly better candidate exists: climb;
  //     h <= tol    =>  certificate that no candidate beats L, even on
  //                     the sampled problem (the continuum has MORE
  //                     constraints, so certainly not there either).
  // On feasibility we adopt the witness, let DC polish it
  // quadratically, and double the kick (geometric climb).  On a
  // certified infeasibility L becomes an upper bound and the kick
  // halves down to a ~1% gap.  `prune_below` carries the best M of the
  // other sign combinations: a combination that cannot beat it is
  // abandoned after a single infeasible probe.
  // ------------------------------------------------------------------
  auto kick0 = [&](double m) {
    return std::max(0.05 * (1.0 + std::abs(m)), 10.0 * spec.tol);
  };
  auto kick_min = [&](double m) {
    return std::max(0.01 * (1.0 + std::abs(m)), 10.0 * spec.tol);
  };
  best.M_upper = std::numeric_limits<double>::infinity();
  // When no dense-valid candidate emerged from DC, fall back to the raw
  // init seed for the probe level and reference.
  double floor = std::max(best.M, prune_below);
  if (!std::isfinite(floor)) floor = M_km1;
  double kick = kick0(floor);
  int stalls = 0;
  for (int probe = 0; probe < spec.max_kick_probes; ++probe) {
    double L = floor + kick;
    // Probe ladder: both normalisations of h (relative units with
    // F_ref = current best F, the eq. 14 denominator, and absolute units
    // rhs = 1) x two Lobatto grid densities (x the two sigma_J mirrors
    // inside probe_lp_robust).  Each axis decorrelates a failure mode
    // that has been observed to produce a false "infeasible" on this
    // ill-conditioned LP: the h normalisation changes the objective
    // scaling, the grid density changes the near-dependent row
    // structure.  The first clearly feasible witness wins; a
    // certificate requires every rung to agree.
    const std::vector<double>* F_norm = best.F.empty() ? &F_km1 : &best.F;
    const int bs_alt = std::max(12, (2 * spec.base_samples) / 3);
    const std::pair<const std::vector<double>*, int> rungs[] = {
        {F_norm, spec.base_samples},
        {nullptr, spec.base_samples},
        {F_norm, bs_alt},
        {nullptr, bs_alt},
    };
    double h_t = -std::numeric_limits<double>::infinity();
    std::vector<double> F_t, P_t;
    for (const auto& [F_ref_v, bs_v] : rungs) {
      auto [h_v, F_v, P_v] = probe_lp_robust(spec, pb, sb, sI, sJ, L,
                                             F_ref_v, psi_I, psi_J,
                                             spec.max_exchange, bs_v);
      if (!F_v.empty() && (F_t.empty() || h_v > h_t)) {
        h_t = h_v; F_t = std::move(F_v); P_t = std::move(P_v);
      }
      if (!F_t.empty() && h_t > spec.tol) break;
    }
    if (!F_t.empty() && h_t > spec.tol) {
      double M_t = min_signed_ratio_minus_psi(F_t, P_t, sb, sJ, psi_J,
                                              spec.refine_samples);
      dc_iterate(spec, pb, sb, sI, sJ, psi_I, psi_J,
                 std::move(F_t), std::move(P_t), M_t, best);
      if (best.M > best.M_upper) {
        // The climb just contradicted an earlier "certificate": that
        // bound was solver noise.  Discard it.
        best.M_upper = std::numeric_limits<double>::infinity();
      }
      if (zolo_trace())
        std::fprintf(stderr, "    kick L=%.5f: feasible, climbed to %.5f\n",
                     L, best.M);
      double new_floor = std::max(best.M, prune_below);
      if (!std::isfinite(new_floor)) new_floor = L;
      // Repeated feasible probes without any dense improvement mean the
      // witnesses cannot be realised on the continuum at this
      // resolution -- stop instead of oscillating.
      stalls = (new_floor <= floor + spec.tol) ? stalls + 1 : 0;
      if (stalls >= 3) break;
      floor = new_floor;
      kick = std::max(2.0 * kick, kick0(floor));
    } else {
      if (!F_t.empty()) {
        // Genuine h <= tol: certified -- no candidate beats L.  (An
        // empty F_t means the LP failed outright: no certificate.)
        best.M_upper = std::min(best.M_upper, L);
        if (zolo_trace())
          std::fprintf(stderr, "    kick L=%.5f: infeasible (bound)\n", L);
      } else if (zolo_trace()) {
        std::fprintf(stderr, "    kick L=%.5f: LP failed\n", L);
      }
      if (kick <= kick_min(floor) * (1.0 + 1e-9)) break;
      kick = std::max(0.5 * kick, kick_min(floor));
    }
  }
  // A bound below the dense-verified achievement is solver noise.
  best.M_upper = std::max(best.M_upper, best.M);
  best.success = !best.F.empty();
  return best;
}

// ---------------------------------------------------------------------------
// Alternation certificate (paper §IV.B)
// ---------------------------------------------------------------------------
//
// Port of the Python reference's extreme_points() + alternation_length():
// detect the near-touch points of the residual against the active
// constraint on each band, then count the longest alternating sign run.
// An optimal characteristic function must alternate on >= nF + nP + 2
// points across I u J.

static void compute_alternation(const Spec& spec, Result& out,
                                const std::vector<int>& sigma_I,
                                const std::vector<int>& sigma_J) {
  const int n_dense = 4000;
  const double tol = 1e-2;
  struct Pt { double w; int sign; };
  std::vector<Pt> pts;

  auto scan = [&](double a, double b,
                  const std::function<double(double)>& resid_fn,
                  const std::function<double(double)>& tol_fn, int which) {
    std::vector<double> abs_r(n_dense);
    std::vector<double> ws(n_dense);
    for (int i = 0; i < n_dense; ++i) {
      ws[i] = a + (b - a) * i / (n_dense - 1);
      abs_r[i] = std::abs(resid_fn(ws[i]));
    }
    for (int i = 1; i + 1 < n_dense; ++i) {
      if (abs_r[i] <= abs_r[i - 1] && abs_r[i] <= abs_r[i + 1] &&
          abs_r[i] < tol_fn(ws[i])) {
        pts.push_back({ws[i], which});
      }
    }
  };

  auto D_of = [&](double w) {
    return polyval(out.F_mono, w) / polyval(out.P_mono, w);
  };

  for (size_t k = 0; k < spec.passbands.size(); ++k) {
    double a = spec.passbands[k].a, b = spec.passbands[k].b;
    auto flat_tol = [&](double) { return tol; };
    scan(a, b, [&](double w) { return D_of(w) - spec.psi_I(w); }, flat_tol, +1);
    scan(a, b, [&](double w) { return D_of(w) + spec.psi_I(w); }, flat_tol, -1);
  }
  for (size_t k = 0; k < spec.stopbands.size(); ++k) {
    double a = spec.stopbands[k].a, b = spec.stopbands[k].b;
    int sig = sigma_J[k];
    auto target = [&](double w) {
      return out.M * stopband_weight(spec.psi_J(w));
    };
    auto rel_tol = [&](double w) {
      return tol * std::max(1.0, std::abs(target(w)));
    };
    if (sig == +1) {
      scan(a, b, [&](double w) { return D_of(w) - target(w); }, rel_tol, -1);
    } else {
      scan(a, b, [&](double w) { return D_of(w) + target(w); }, rel_tol, +1);
    }
  }
  (void)sigma_I;

  std::sort(pts.begin(), pts.end(),
            [](const Pt& x, const Pt& y) { return x.w < y.w; });
  std::vector<Pt> dedup;
  for (const auto& p : pts) {
    if (!dedup.empty() && std::abs(p.w - dedup.back().w) < 1e-4) continue;
    dedup.push_back(p);
  }

  // Longest alternating subsequence of signs.
  int run = 0, best = 0, prev = 0;
  for (const auto& p : dedup) {
    if (run == 0 || p.sign != prev) {
      run += 1;
      best = std::max(best, run);
    } else {
      run = 1;
    }
    prev = p.sign;
  }

  out.alt_count = best;
  out.alt_required = spec.nF + spec.nP + 2;
  out.alt_omega.clear();
  out.alt_sign.clear();
  for (const auto& p : dedup) {
    out.alt_omega.push_back(p.w);
    out.alt_sign.push_back(p.sign);
  }
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

Result solve_zolotarev(const Spec& spec) {
  Result out;
  if (spec.passbands.empty() || spec.stopbands.empty() || spec.nF < 1) {
    out.message = "empty spec";
    return out;
  }

  // Rescale omega to [-1, 1] for numerical conditioning.
  double scale = 1.0;
  if (spec.rescale) {
    for (const auto& iv : spec.passbands) scale = std::max(scale, std::max(std::abs(iv.a), std::abs(iv.b)));
    for (const auto& iv : spec.stopbands) scale = std::max(scale, std::max(std::abs(iv.a), std::abs(iv.b)));
  }
  std::vector<Interval> pb_s, sb_s;
  for (const auto& iv : spec.passbands) pb_s.push_back({iv.a / scale, iv.b / scale});
  for (const auto& iv : spec.stopbands) sb_s.push_back({iv.a / scale, iv.b / scale});

  const PsiFn orig_psi_I = spec.psi_I;
  const PsiFn orig_psi_J = spec.psi_J;
  PsiFn psi_I = [orig_psi_I, scale](double w) { return orig_psi_I(w * scale); };
  PsiFn psi_J = [orig_psi_J, scale](double w) { return orig_psi_J(w * scale); };

  const int r = static_cast<int>(pb_s.size());
  const int p = static_cast<int>(sb_s.size());

  SignedResult best_signed;
  std::vector<int> best_sI, best_sJ;
  double global_upper = -std::numeric_limits<double>::infinity();

  for (int biI = 0; biI < (1 << r); ++biI) {
    std::vector<int> sI(r);
    for (int k = 0; k < r; ++k) sI[k] = (biI & (1 << k)) ? -1 : 1;
    if (sI[0] != 1) continue;   // kill the (F,P) -> (-F,-P) symmetry

    for (int biJ = 0; biJ < (1 << p); ++biJ) {
      std::vector<int> sJ(p);
      for (int k = 0; k < p; ++k) sJ[k] = (biJ & (1 << k)) ? -1 : 1;
      // F -> -F (P unchanged) flips every sigma_J while leaving the
      // passband constraints invariant, so (sigma_I, sigma_J) and
      // (sigma_I, -sigma_J) are the same subproblem; probe_lp_robust
      // exploits the redundancy per LP solve instead.
      if (sJ[0] != 1) continue;

      if (zolo_trace()) {
        std::fprintf(stderr, "sigma_I=[");
        for (int x : sI) std::fprintf(stderr, "%d ", x);
        std::fprintf(stderr, "]  sigma_J=[");
        for (int x : sJ) std::fprintf(stderr, "%d ", x);
        std::fprintf(stderr, "]\n");
      }
      auto r1 = solve_signed(spec, pb_s, sb_s, sI, sJ, psi_I, psi_J,
                             best_signed.M);
      if (zolo_trace())
        std::fprintf(stderr, "  -> success=%d M=%.5f (<= %.5f)\n",
                     r1.success ? 1 : 0, r1.M, r1.M_upper);
      global_upper = std::max(global_upper, r1.M_upper);
      if (!r1.success) continue;
      if (r1.M > best_signed.M) {
        best_signed = std::move(r1);
        best_sI = sI;
        best_sJ = sJ;
      }
    }
  }

  if (!best_signed.success) {
    out.message = "no sign combination produced a feasible filter";
    return out;
  }

  // Convert Chebyshev coeffs (in the scaled variable t=omega/scale) back to
  // monomial coefficients in omega.
  std::vector<double> F_mon_t = cheb_to_mono(best_signed.F);
  std::vector<double> P_mon_t = cheb_to_mono(best_signed.P);
  out.F_mono = rescale_mono(F_mon_t, 1.0 / scale);
  out.P_mono = rescale_mono(P_mon_t, 1.0 / scale);
  out.success = true;
  out.message = "ok";
  out.M = best_signed.M;
  out.M_upper = std::max(global_upper, best_signed.M);
  out.sigma_I = best_sI;
  out.sigma_J = best_sJ;
  out.scale = scale;
  compute_alternation(spec, out, best_sI, best_sJ);
  return out;
}

Response scattering_response(const std::vector<double>& F,
                             const std::vector<double>& P,
                             const std::vector<double>& omega) {
  Response r;
  r.omega = omega;
  r.S21_sq.resize(omega.size());
  r.S11_sq.resize(omega.size());
  for (size_t i = 0; i < omega.size(); ++i) {
    double Fv = polyval(F, omega[i]);
    double Pv = polyval(P, omega[i]);
    double D = Fv / Pv;
    double D2 = D * D;
    r.S21_sq[i] = 1.0 / (1.0 + D2);
    r.S11_sq[i] = D2 / (1.0 + D2);
  }
  return r;
}

double rl_db_to_psi(double rl_db) {
  double r = std::pow(10.0, -rl_db / 20.0);
  return r / std::sqrt(1.0 - r * r);
}
double rej_db_to_psi(double rej_db) {
  return std::sqrt(std::pow(10.0, rej_db / 10.0) - 1.0);
}

}  // namespace zolo
