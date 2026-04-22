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

// Chebyshev nodes of the first kind in [a, b].
static std::vector<double> chebyshev_nodes(double a, double b, int n) {
  std::vector<double> out(n);
  for (int k = 0; k < n; ++k) {
    double c = std::cos((2.0 * k + 1.0) * M_PI / (2.0 * n));
    out[k] = 0.5 * (a + b) + 0.5 * (b - a) * c;
  }
  return out;
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

static LPResult build_and_solve_lp(const std::vector<SamplePt>& I_pts,
                                   const std::vector<SamplePt>& J_pts,
                                   int nF, int nP, double M,
                                   const PsiFn& psi_I, const PsiFn& psi_J,
                                   double coef_bound,
                                   const std::vector<double>* F_ref) {
  const int n_f = nF + 1;
  const int n_p = nP;           // coefficient of T_nP fixed to 1
  const int nvar = n_f + n_p + 1;
  const int h_idx = nvar - 1;

  // Build column-compressed matrix incrementally as row triplets.
  std::vector<double> col_lower(nvar, -coef_bound);
  std::vector<double> col_upper(nvar, +coef_bound);
  col_lower[h_idx] = -kHighsInf;   // h free
  col_upper[h_idx] = +kHighsInf;

  std::vector<double> col_cost(nvar, 0.0);
  col_cost[h_idx] = -1.0;          // maximise h => minimise -h

  // Assemble rows in LIL form then convert.
  struct Row { std::vector<int> idx; std::vector<double> val; double ub; };
  std::vector<Row> rows;
  rows.reserve((2 * J_pts.size()) + (3 * I_pts.size()));

  std::vector<double> Fvec, Pvec_all;

  // Stopband rows.
  for (const auto& s : J_pts) {
    double y = s.x;
    int sig = s.sigma;
    double psi = psi_J(y);
    double Meff = psi + M;
    cheb_basis(y, n_f, Fvec);
    cheb_basis(y, nP + 1, Pvec_all);
    double P_lead = Pvec_all[nP];
    double rhs = 1.0;
    if (F_ref != nullptr) {
      rhs = std::max(std::abs(cheb_eval(*F_ref, y)), 1e-12);
    }
    // Row A:  -sig Fvec.f  +  Meff Pvec_low.p  +  rhs*h  <=  -Meff P_lead
    {
      Row r;
      for (int i = 0; i < n_f; ++i) {
        r.idx.push_back(i);
        r.val.push_back(-sig * Fvec[i]);
      }
      for (int i = 0; i < n_p; ++i) {
        r.idx.push_back(n_f + i);
        r.val.push_back(Meff * Pvec_all[i]);
      }
      r.idx.push_back(h_idx);
      r.val.push_back(rhs);
      r.ub = -Meff * P_lead;
      rows.push_back(std::move(r));
    }
    // Row B:  -sig Fvec.f  -  Meff Pvec_low.p  +  rhs*h  <=  +Meff P_lead
    {
      Row r;
      for (int i = 0; i < n_f; ++i) {
        r.idx.push_back(i);
        r.val.push_back(-sig * Fvec[i]);
      }
      for (int i = 0; i < n_p; ++i) {
        r.idx.push_back(n_f + i);
        r.val.push_back(-Meff * Pvec_all[i]);
      }
      r.idx.push_back(h_idx);
      r.val.push_back(rhs);
      r.ub = Meff * P_lead;
      rows.push_back(std::move(r));
    }
  }

  // Passband rows: sigma P >= 0,  F <= sig*psi*P,  -F <= sig*psi*P.
  for (const auto& s : I_pts) {
    double x = s.x;
    int sig = s.sigma;
    double psi = psi_I(x);
    cheb_basis(x, n_f, Fvec);
    cheb_basis(x, nP + 1, Pvec_all);
    double P_lead = Pvec_all[nP];

    // sig*P >= 0  =>  -sig Pvec_low.p  <=  sig*P_lead
    {
      Row r;
      for (int i = 0; i < n_p; ++i) {
        r.idx.push_back(n_f + i);
        r.val.push_back(-sig * Pvec_all[i]);
      }
      r.ub = sig * P_lead;
      rows.push_back(std::move(r));
    }
    // F - sig*psi*P <= 0  =>  Fvec.f - sig*psi Pvec_low.p  <=  sig*psi P_lead
    {
      Row r;
      for (int i = 0; i < n_f; ++i) {
        r.idx.push_back(i);
        r.val.push_back(Fvec[i]);
      }
      for (int i = 0; i < n_p; ++i) {
        r.idx.push_back(n_f + i);
        r.val.push_back(-sig * psi * Pvec_all[i]);
      }
      r.ub = sig * psi * P_lead;
      rows.push_back(std::move(r));
    }
    // -F - sig*psi*P <= 0
    {
      Row r;
      for (int i = 0; i < n_f; ++i) {
        r.idx.push_back(i);
        r.val.push_back(-Fvec[i]);
      }
      for (int i = 0; i < n_p; ++i) {
        r.idx.push_back(n_f + i);
        r.val.push_back(-sig * psi * Pvec_all[i]);
      }
      r.ub = sig * psi * P_lead;
      rows.push_back(std::move(r));
    }
  }

  // Use HiGHS's incremental addCol/addRow API.  When building the matrix
  // by hand and submitting via passModel(), HiGHS was reading the sparse
  // matrix in a way that misinterpreted either the format or start-array
  // layout and returned spurious infeasible statuses.  The incremental
  // API is both safer and closer to how scipy.optimize.linprog does it.
  Highs highs;
  highs.setOptionValue("output_flag", false);

  for (int j = 0; j < nvar; ++j) {
    HighsStatus rc = highs.addCol(col_cost[j], col_lower[j], col_upper[j],
                                  0, nullptr, nullptr);
    if (rc != HighsStatus::kOk) return {false, 0, {}, {}};
  }
  highs.changeObjectiveSense(ObjSense::kMinimize);

  for (const auto& r : rows) {
    std::vector<HighsInt> idx(r.idx.size());
    for (size_t k = 0; k < r.idx.size(); ++k) idx[k] = r.idx[k];
    HighsStatus rc = highs.addRow(-kHighsInf, r.ub,
                                  static_cast<HighsInt>(idx.size()),
                                  idx.data(), r.val.data());
    if (rc != HighsStatus::kOk) return {false, 0, {}, {}};
  }

  HighsStatus st = HighsStatus::kOk;
  st = highs.run();
  if (st != HighsStatus::kOk) {
    if (zolo_trace()) std::fprintf(stderr, "  LP run failed (status=%d)\n",
                                   static_cast<int>(st));
    return {false, 0, {}, {}};
  }
  HighsModelStatus ms = highs.getModelStatus();
  if (ms != HighsModelStatus::kOptimal) {
    if (zolo_trace())
      std::fprintf(stderr, "  LP model status = %d\n",
                   static_cast<int>(ms));
    return {false, 0, {}, {}};
  }

  const auto& sol = highs.getSolution();
  LPResult out;
  out.success = true;
  out.h = sol.col_value[h_idx];
  out.F.assign(sol.col_value.begin(), sol.col_value.begin() + n_f);
  out.P.assign(n_p + 1, 0.0);
  for (int i = 0; i < n_p; ++i) out.P[i] = sol.col_value[n_f + i];
  out.P[n_p] = 1.0;   // pinned leading Chebyshev coefficient
  return out;
}

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
         const PsiFn& psi_I, const PsiFn& psi_J) {
  std::vector<SamplePt> I_pts, J_pts;
  for (size_t k = 0; k < passbands.size(); ++k) {
    auto xs = chebyshev_nodes(passbands[k].a, passbands[k].b,
                              std::max(6, spec.base_samples));
    for (double x : xs) I_pts.push_back({x, sigma_I[k]});
  }
  for (size_t k = 0; k < stopbands.size(); ++k) {
    double w = std::abs(stopbands[k].b - stopbands[k].a);
    int n = static_cast<int>(std::ceil(
        spec.base_samples * std::max(1.0, 0.5 + std::log10(1.0 + 10.0 * w))));
    auto ys = chebyshev_nodes(stopbands[k].a, stopbands[k].b, n);
    for (double y : ys) J_pts.push_back({y, sigma_J[k]});
  }

  double h = -std::numeric_limits<double>::infinity();
  std::vector<double> F, P;

  for (int it = 0; it < spec.max_exchange; ++it) {
    auto lp = build_and_solve_lp(I_pts, J_pts, spec.nF, spec.nP, M,
                                 psi_I, psi_J, spec.coef_bound, F_ref);
    if (!lp.success) return {h, F, P};
    h = lp.h;
    F = std::move(lp.F);
    P = std::move(lp.P);

    // Dense verification.
    bool added = false;
    auto add_worst = [&](std::vector<SamplePt>& set, double x, int sig) {
      set.push_back({x, sig}); added = true;
    };

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
        double vr = std::abs(Fx) - psi * std::abs(Px);
        double vs = -sig * Px;
        if (vr > best_ratio) { best_ratio = vr; best_ratio_x = x; }
        if (vs > best_sig)   { best_sig = vs;   best_sig_x = x; }
      }
      if (best_ratio > spec.tol) add_worst(I_pts, best_ratio_x, sig);
      if (best_sig   > spec.tol) add_worst(I_pts, best_sig_x, sig);
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
        double psi = psi_J(y);
        double Meff = psi + M;
        double rhs = (F_ref ? std::abs(cheb_eval(*F_ref, y)) : 1.0);
        double v = h * rhs - (sig * Fy - Meff * std::abs(Py));
        if (v > best_v) { best_v = v; best_v_x = y; }
      }
      if (best_v > spec.tol * std::max(1.0, std::abs(h)))
        add_worst(J_pts, best_v_x, sig);
    }

    if (!added) break;
  }
  return {h, F, P};
}

// ---------------------------------------------------------------------------
// Signed optimum via iterative differential correction
// ---------------------------------------------------------------------------

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
      double r = sig * Fy / std::abs(Py) - psi_J(y);
      if (r < best) best = r;
    }
  }
  return best;
}

struct SignedResult {
  bool success = false;
  double M = -std::numeric_limits<double>::infinity();
  std::vector<double> F, P;  // Cheb
};

static SignedResult solve_signed(const Spec& spec,
                                 const std::vector<Interval>& pb,
                                 const std::vector<Interval>& sb,
                                 const std::vector<int>& sI,
                                 const std::vector<int>& sJ,
                                 const PsiFn& psi_I,
                                 const PsiFn& psi_J) {
  // Step 0: probe at a guaranteed-feasible M_{-1}.
  double psi_J_max = 0.0;
  for (const auto& iv : sb) {
    for (int i = 0; i < 11; ++i) {
      double w = iv.a + (iv.b - iv.a) * i / 10.0;
      psi_J_max = std::max(psi_J_max, psi_J(w));
    }
  }
  double M_start = -psi_J_max - 1.0;

  auto [h0, F_km1, P_km1] = probe_lp(spec, pb, sb, sI, sJ, M_start, nullptr,
                                     psi_I, psi_J);
  if (F_km1.empty()) {
    if (zolo_trace()) std::fprintf(stderr, "    init probe failed\n");
    return {};
  }
  double M_km1 = min_signed_ratio_minus_psi(F_km1, P_km1, sb, sJ, psi_J,
                                            spec.refine_samples);
  if (zolo_trace())
    std::fprintf(stderr, "    init M_start=%.3f h0=%.3e M_0=%.5f\n",
                 M_start, h0, M_km1);

  SignedResult best;
  best.success = true;
  best.M = M_km1;
  best.F = F_km1;
  best.P = P_km1;

  for (int k = 1; k <= spec.max_iter; ++k) {
    auto [h, F_k, P_k] = probe_lp(spec, pb, sb, sI, sJ, M_km1, &F_km1,
                                  psi_I, psi_J);
    if (zolo_trace())
      std::fprintf(stderr, "    iter %d: h=%.3e\n", k, h);
    if (F_k.empty() || h <= spec.tol) break;
    double M_k = min_signed_ratio_minus_psi(F_k, P_k, sb, sJ, psi_J,
                                            spec.refine_samples);
    if (zolo_trace())
      std::fprintf(stderr, "            M=%.5f  dM=%.2e\n",
                   M_k, M_k - M_km1);
    if (M_k > best.M) {
      best.M = M_k; best.F = F_k; best.P = P_k;
    }
    if (std::abs(M_k - M_km1) < spec.tol * std::max(1.0, std::abs(M_k))) break;
    F_km1 = std::move(F_k);
    P_km1 = std::move(P_k);
    M_km1 = M_k;
  }
  return best;
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

  for (int biI = 0; biI < (1 << r); ++biI) {
    std::vector<int> sI(r);
    for (int k = 0; k < r; ++k) sI[k] = (biI & (1 << k)) ? -1 : 1;
    if (sI[0] != 1) continue;   // kill the (F,P) -> (-F,-P) symmetry

    for (int biJ = 0; biJ < (1 << p); ++biJ) {
      std::vector<int> sJ(p);
      for (int k = 0; k < p; ++k) sJ[k] = (biJ & (1 << k)) ? -1 : 1;

      if (zolo_trace()) {
        std::fprintf(stderr, "sigma_I=[");
        for (int x : sI) std::fprintf(stderr, "%d ", x);
        std::fprintf(stderr, "]  sigma_J=[");
        for (int x : sJ) std::fprintf(stderr, "%d ", x);
        std::fprintf(stderr, "]\n");
      }
      auto r1 = solve_signed(spec, pb_s, sb_s, sI, sJ, psi_I, psi_J);
      if (zolo_trace())
        std::fprintf(stderr, "  -> success=%d M=%.5f\n",
                     r1.success ? 1 : 0, r1.M);
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
  out.sigma_I = best_sI;
  out.sigma_J = best_sJ;
  out.scale = scale;
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
