// Self-contained two-phase simplex LP solver.
//
// Solves
//     minimize   c . x
//     subject to A x <= b,   x_i in [lower_i, upper_i]
//
// with  +/- std::numeric_limits<double>::infinity()  accepted for the
// column bounds.  Variables with a bounded lower edge are shifted to make
// the lower edge zero; variables bounded only from above are sign-flipped;
// variables free on both sides are split into positive and negative parts.
//
// The core two-phase simplex is adapted from the gpt5.5 exploration:
// dense tableau, Bland's rule tie-break on degenerate pivots, automatic
// row and objective scaling, plus an iteration cap so pathological inputs
// return Status::Unbounded rather than hang.  ~260 lines, no external deps.
//
// IMPORTANT -- when NOT to use this:
// Our multiband synthesis LPs sample smooth polynomials densely at
// Chebyshev-Lobatto nodes, so adjacent sample rows are nearly parallel
// (the rows differ by Chebyshev polynomial differences that are small).
// The dense-tableau simplex degenerates on these (lots of tied pivot
// ratios, slow progress through redundant vertices) and scales poorly:
// empirically, native benchmarks go from <1 ms at 35 rows to >60 s at
// 70 rows.  For LPs with more than ~50 rows you'll want a revised
// simplex with LU updates or a sparse solver like HiGHS.  mini_lp is
// kept in the tree because its API is clean and it's proven correct on
// small LPs (see mini_lp_test.cpp).
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace mlp {

enum class Status { Optimal, Infeasible, Unbounded };

struct Result {
  Status status = Status::Optimal;
  double objective = 0.0;
  std::vector<double> x;
  std::string message;
};

namespace detail {

constexpr double kEps = 1e-9;
constexpr double kInf = std::numeric_limits<double>::infinity();

// Two-phase simplex on a dense tableau in standard form
//     min c.x  s.t.  A x <= b,  x >= 0.
// Adapted from the gpt5.5 implementation (MIT-licensed).
class TwoPhaseSimplex {
 public:
  TwoPhaseSimplex(const std::vector<std::vector<double>>& A,
                  const std::vector<double>& b,
                  const std::vector<double>& c)
      : m_(static_cast<int>(b.size())),
        n_(static_cast<int>(c.size())),
        basis_(m_),
        non_basis_(n_ + 1),
        t_(m_ + 2, std::vector<double>(n_ + 2, 0.0)) {
    for (int i = 0; i < m_; ++i)
      for (int j = 0; j < n_; ++j) t_[i][j] = A[i][j];
    for (int i = 0; i < m_; ++i) {
      basis_[i] = n_ + i;
      t_[i][n_] = -1.0;
      t_[i][n_ + 1] = b[i];
    }
    for (int j = 0; j < n_; ++j) {
      non_basis_[j] = j;
      t_[m_][j] = -c[j];
    }
    non_basis_[n_] = -1;
    t_[m_ + 1][n_] = 1.0;
  }

  Result solve() {
    Result r;
    if (n_ == 0) { r.status = Status::Optimal; return r; }

    int row = 0;
    for (int i = 1; i < m_; ++i)
      if (t_[i][n_ + 1] < t_[row][n_ + 1]) row = i;

    if (m_ > 0 && t_[row][n_ + 1] < -kEps) {
      pivot(row, n_);
      if (!simplex(1) || t_[m_ + 1][n_ + 1] < -kEps ||
          std::abs(t_[m_ + 1][n_ + 1]) > 1e-7) {
        r.status = Status::Infeasible;
        r.message = "linear program is infeasible";
        return r;
      }
      for (int i = 0; i < m_; ++i) {
        if (basis_[i] == -1) {
          int s = -1;
          for (int j = 0; j <= n_; ++j)
            if (s == -1 || non_basis_[j] < non_basis_[s]) s = j;
          pivot(i, s);
        }
      }
    }

    if (!simplex(0)) {
      r.status = Status::Unbounded;
      r.message = "linear program is unbounded";
      return r;
    }
    r.status = Status::Optimal;
    r.objective = t_[m_][n_ + 1];
    r.x.assign(n_, 0.0);
    for (int i = 0; i < m_; ++i)
      if (basis_[i] < n_) r.x[basis_[i]] = t_[i][n_ + 1];
    return r;
  }

 private:
  void pivot(int r, int s) {
    const double inv = 1.0 / t_[r][s];
    for (int i = 0; i < m_ + 2; ++i) {
      if (i == r) continue;
      for (int j = 0; j < n_ + 2; ++j) {
        if (j == s) continue;
        t_[i][j] -= t_[r][j] * t_[i][s] * inv;
      }
    }
    for (int j = 0; j < n_ + 2; ++j) if (j != s) t_[r][j] *= inv;
    for (int i = 0; i < m_ + 2; ++i) if (i != r) t_[i][s] *= -inv;
    t_[r][s] = inv;
    std::swap(basis_[r], non_basis_[s]);
  }

  // Returns true on optimal, false on unbounded or iteration cap.  The
  // iteration cap is generous (200*(m+n)) but prevents indefinite hangs
  // on degenerate LPs that Bland's rule alone can't escape quickly.
  bool simplex(int phase) {
    const int obj = phase == 1 ? m_ + 1 : m_;
    const int pivot_cap = 200 * (m_ + n_);
    for (int it = 0; it < pivot_cap; ++it) {
      int s = -1;
      for (int j = 0; j <= n_; ++j) {
        if (phase == 0 && non_basis_[j] == -1) continue;
        if (s == -1 || t_[obj][j] < t_[obj][s] - kEps ||
            (std::abs(t_[obj][j] - t_[obj][s]) <= kEps &&
             non_basis_[j] < non_basis_[s])) s = j;
      }
      if (t_[obj][s] >= -kEps) return true;

      int r = -1;
      for (int i = 0; i < m_; ++i) {
        if (t_[i][s] <= kEps) continue;
        if (r == -1) { r = i; continue; }
        const double lhs = t_[i][n_ + 1] / t_[i][s];
        const double rhs = t_[r][n_ + 1] / t_[r][s];
        if (lhs < rhs - kEps ||
            (std::abs(lhs - rhs) <= kEps && basis_[i] < basis_[r])) r = i;
      }
      if (r == -1) return false;
      pivot(r, s);
    }
    return false;  // iteration cap hit -- treat as "unbounded/stalled"
  }

  int m_, n_;
  std::vector<int> basis_;
  std::vector<int> non_basis_;
  std::vector<std::vector<double>> t_;
};

// Per-row and objective scaling.  Keeps the simplex numerically sane when
// LP coefficients span many orders of magnitude (which happens with high
// degree polynomials sampled at both small and large omegas).
//
// TwoPhaseSimplex internally solves   max c . x  (standard textbook form)
// but our public API is  minimise c . x  -- so we pass -c, solve max(-c.x),
// then negate the returned objective to get min(c.x).
inline Result solve_standard(std::vector<std::vector<double>> A,
                             std::vector<double> b,
                             std::vector<double> c) {
  for (std::size_t i = 0; i < A.size(); ++i) {
    double s = std::max(1.0, std::abs(b[i]));
    for (double v : A[i]) s = std::max(s, std::abs(v));
    for (double& v : A[i]) v /= s;
    b[i] /= s;
  }
  double os = 1.0;
  for (double v : c) os = std::max(os, std::abs(v));
  std::vector<double> c_neg(c.size());
  for (std::size_t i = 0; i < c.size(); ++i) c_neg[i] = -c[i] / os;
  TwoPhaseSimplex lp(A, b, c_neg);
  auto r = lp.solve();
  r.objective *= -os;  // min c.x  =  -max(-c.x)
  return r;
}

}  // namespace detail

// Boxed-LP wrapper:  minimize c'x  s.t.  A x <= b,  lower_i <= x_i <= upper_i.
// Accepts  +/- infinity for the bounds.
inline Result solve(const std::vector<double>& c,
                    const std::vector<double>& lower,
                    const std::vector<double>& upper,
                    const std::vector<std::vector<double>>& A,
                    const std::vector<double>& b) {
  const int n = static_cast<int>(c.size());
  const int m_in = static_cast<int>(b.size());
  if (lower.size() != static_cast<std::size_t>(n) ||
      upper.size() != static_cast<std::size_t>(n)) {
    Result r; r.status = Status::Infeasible;
    r.message = "bounds/cost dimension mismatch"; return r;
  }

  // Pass 1: decide per-column transform and count extra columns/rows.
  //   T[i] = 0: x_i = lower_i + x'_i,   x'_i >= 0, cap row if upper finite
  //   T[i] = 1: x_i = upper_i - x'_i,   x'_i >= 0  (lower = -inf, upper finite)
  //   T[i] = 2: x_i = x+ - x-,          both >= 0  (both infinite)
  //   T[i] = 3: x_i = lower_i + x'_i - x''_i  (both finite via shift, or
  //                we just cap x')  -- we use T[0] with upper cap here.
  // We emit (in order): standard columns, plus extras for T=2 splits.
  // For T=0 with finite upper, we add a cap row x' <= upper - lower.
  std::vector<int> T(n, 0);
  std::vector<double> offset(n, 0.0);      // contribution l_i or u_i to x_i
  std::vector<double> sign(n, 1.0);        // x_i = offset + sign * x'_i (+ ...)
  std::vector<double> cap(n, detail::kInf);   // upper - lower if both finite
  int extra_split = 0;                     // # of T=2 variables (need 2 cols)
  int extra_cap   = 0;                     // # of T=0 with finite upper
  for (int j = 0; j < n; ++j) {
    const bool lfin = std::isfinite(lower[j]);
    const bool ufin = std::isfinite(upper[j]);
    if (lfin && ufin) {
      T[j] = 0; offset[j] = lower[j]; sign[j] = 1.0;
      cap[j] = upper[j] - lower[j];
      if (cap[j] < -detail::kEps) {
        Result r; r.status = Status::Infeasible;
        r.message = "empty interval"; return r;
      }
      ++extra_cap;
    } else if (!lfin && ufin) {
      T[j] = 1; offset[j] = upper[j]; sign[j] = -1.0;
    } else if (lfin && !ufin) {
      T[j] = 0; offset[j] = lower[j]; sign[j] = 1.0;   // no cap row
    } else {
      T[j] = 2; offset[j] = 0.0; sign[j] = 1.0;
      ++extra_split;
    }
  }

  // Build standard-form LP with:
  //   columns = n + extra_split      (the second half of split variables)
  //   rows    = m_in + extra_cap
  const int n_std = n + extra_split;
  const int m_std = m_in + extra_cap;

  // Allocate.
  std::vector<double> c_std(n_std, 0.0);
  std::vector<std::vector<double>> A_std(m_std, std::vector<double>(n_std, 0.0));
  std::vector<double> b_std(m_std, 0.0);

  // Fill c.  For T=0 and T=1, cost coef is sign[j]*c[j].  For T=2, cost is c[j]
  // on the "+" part and -c[j] on the "-" part.
  int split_col = n;
  std::vector<int> split_neg_col(n, -1);
  for (int j = 0; j < n; ++j) {
    if (T[j] == 2) {
      c_std[j] = c[j];
      c_std[split_col] = -c[j];
      split_neg_col[j] = split_col;
      ++split_col;
    } else {
      c_std[j] = sign[j] * c[j];
    }
  }

  // Transform each original A row.  For T=0 / T=1: A'_ij = sign[j]*A_ij.
  // For T=2: A'_ij = A_ij on the + column, -A_ij on the - column.
  // RHS becomes b_i - sum_j (A_ij * offset[j]).
  for (int i = 0; i < m_in; ++i) {
    double bi = b[i];
    for (int j = 0; j < n; ++j) {
      bi -= A[i][j] * offset[j];
      if (T[j] == 2) {
        A_std[i][j] = A[i][j];
        A_std[i][split_neg_col[j]] = -A[i][j];
      } else {
        A_std[i][j] = sign[j] * A[i][j];
      }
    }
    b_std[i] = bi;
  }

  // Add cap rows  x'_j <= upper_j - lower_j  for T=0 with finite upper.
  int cap_row = m_in;
  for (int j = 0; j < n; ++j) {
    if (T[j] == 0 && std::isfinite(cap[j])) {
      A_std[cap_row][j] = 1.0;
      b_std[cap_row] = cap[j];
      ++cap_row;
    }
  }

  // Solve.
  auto std_res = detail::solve_standard(std::move(A_std), std::move(b_std),
                                        std::move(c_std));
  Result out;
  out.status = std_res.status;
  out.objective = std_res.objective;
  out.message = std_res.message;
  if (std_res.status != Status::Optimal) return out;

  // Back-substitute to the original variables.
  out.x.assign(n, 0.0);
  for (int j = 0; j < n; ++j) {
    double x_prime = std_res.x[j];
    if (T[j] == 2) x_prime -= std_res.x[split_neg_col[j]];
    out.x[j] = offset[j] + sign[j] * x_prime;
  }
  return out;
}

}  // namespace mlp
