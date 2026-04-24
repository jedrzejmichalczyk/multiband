// Bench the dense two-phase simplex on an LP the size of Example 1.
#include "mini_lp.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>

int main() {
  using clock = std::chrono::steady_clock;

  const int nF = 9;
  const int n = nF + 1 + 1;  // f_0..f_9 + h
  const double INF = std::numeric_limits<double>::infinity();
  std::vector<double> c(n, 0.0);
  c[n - 1] = -1.0;  // maximise h
  std::vector<double> lo(n, -1e6), hi(n, 1e6);
  lo[n - 1] = -INF; hi[n - 1] = +INF;

  // Build ~240-row passband + 240-row stopband LP like Example 1.
  std::vector<std::vector<double>> A;
  std::vector<double> b;
  auto add = [&](std::vector<double> r, double ub){ A.push_back(r); b.push_back(ub); };

  auto cheb = [](double x, int N, std::vector<double>& out){
    out.assign(N, 0.0);
    if (N>=1) out[0]=1.0;
    if (N>=2) out[1]=x;
    for(int k=2; k<N; ++k) out[k] = 2*x*out[k-1] - out[k-2];
  };

  // Scaled passbands and stopbands (scale=10) as Lobatto nodes.
  auto lobatto = [](double a, double b, int N){
    std::vector<double> out(N);
    if (N < 2) { out = {0.5*(a+b)}; return out; }
    for (int k=0;k<N;++k){
      double c = std::cos(M_PI*k/(N-1));
      out[k] = 0.5*(a+b) + 0.5*(b-a)*c;
    }
    return out;
  };

  const int NS = 40;
  std::vector<std::pair<double,double>> pb = {{-0.1,-0.0625},{0.025,0.1}};
  std::vector<std::pair<double,double>> sb = {{-1.0,-0.1188},{-0.05,0.0125},{0.1212,1.0}};
  double psi_I = 0.1005;

  std::vector<double> F, P;
  for (auto [a,bb] : pb) {
    for (double x : lobatto(a,bb,NS)) {
      cheb(x, nF+1, F);
      cheb(x, 4, P);
      // F(x) <= psi_I  (no P)
      std::vector<double> row(n, 0);
      for (int i=0;i<nF+1;++i) row[i] = F[i];
      add(row, psi_I);
      for (int i=0;i<nF+1;++i) row[i] = -F[i];
      add(row, psi_I);
    }
  }
  for (auto [a,bb] : sb) {
    for (double y : lobatto(a,bb,NS)) {
      cheb(y, nF+1, F);
      std::vector<double> row(n, 0);
      // sigma F(y) >= h  -> -sigma*F + h <= 0
      for (int i=0;i<nF+1;++i) row[i] = -F[i];
      row[n-1] = 1;
      add(row, 0);
    }
  }

  std::printf("LP size: %zu rows x %d cols\n", A.size(), n);

  auto t0 = clock::now();
  auto r = mlp::solve(c, lo, hi, A, b);
  auto dt = std::chrono::duration<double>(clock::now() - t0).count();
  const char* s = r.status==mlp::Status::Optimal ? "OPT" : "NON-OPT";
  std::printf("  %s  obj=%.4f  in %.3fs\n", s, r.objective, dt);
  return 0;
}
