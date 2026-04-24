// Bench: time a single production-sized LP solve (280 rows x 11 cols)
// in NATIVE mini_lp.  Used to triage the WASM slowdown.
//
// Compile:
//   g++ -std=c++17 -O2 -D_USE_MATH_DEFINES mini_lp_bench.cpp -o mini_lp_bench
#include "mini_lp.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main() {
  using clock = std::chrono::steady_clock;

  const int nF = 9;
  const int n = nF + 1 + 1;  // f_0..f_9 + h
  const double INF = std::numeric_limits<double>::infinity();
  std::vector<double> c(n, 0.0);
  c[n - 1] = -1.0;
  std::vector<double> lo(n, -1e6), hi(n, 1e6);
  lo[n - 1] = -INF; hi[n - 1] = +INF;

  std::vector<std::vector<double>> A;
  std::vector<double> b;
  auto add = [&](std::vector<double> r, double ub){ A.push_back(r); b.push_back(ub); };

  auto cheb = [](double x, int N, std::vector<double>& out){
    out.assign(N, 0.0);
    if (N>=1) out[0]=1.0;
    if (N>=2) out[1]=x;
    for(int k=2; k<N; ++k) out[k] = 2*x*out[k-1] - out[k-2];
  };
  auto lobatto = [](double a, double b, int N){
    std::vector<double> out(N);
    if (N < 2) { out = {0.5*(a+b)}; return out; }
    for (int k=0;k<N;++k){
      double c = std::cos(M_PI*k/(N-1));
      out[k] = 0.5*(a+b) + 0.5*(b-a)*c;
    }
    return out;
  };

  // Run at several sample counts to see where the cliff is.
  for (int NS : {5, 8, 10, 12, 15, 20})
  {
  A.clear(); b.clear();
  std::vector<std::pair<double,double>> pb = {{-1.0,-0.5},{0.5,1.0}};
  std::vector<std::pair<double,double>> sb = {{-2.0,-1.2},{-0.3,0.3},{1.2,2.0}};
  double psi_I = 1.0;   // classical Zolotarev, bound |F|<=|P|=1 trivially
  std::vector<double> F;
  for (auto iv : pb) {
    for (double x : lobatto(iv.first, iv.second, NS)) {
      cheb(x, nF+1, F);
      std::vector<double> row(n, 0);
      for (int i=0;i<nF+1;++i) row[i] = F[i];
      add(row, psi_I);
      for (int i=0;i<nF+1;++i) row[i] = -F[i];
      add(row, psi_I);
    }
  }
  for (auto iv : sb) {
    for (double y : lobatto(iv.first, iv.second, NS)) {
      cheb(y, nF+1, F);
      std::vector<double> row(n, 0);
      // F(y) >= h  for sigma_J = +1
      for (int i=0;i<nF+1;++i) row[i] = -F[i];
      row[n-1] = 1;
      add(row, 0);
    }
  }

  auto t0 = clock::now();
  auto r = mlp::solve(c, lo, hi, A, b);
  auto dt = std::chrono::duration<double>(clock::now() - t0).count();
  const char* s = r.status==mlp::Status::Optimal ? "OPT"
                 : r.status==mlp::Status::Infeasible ? "INFEAS" : "UNBND";
  std::printf("NS=%d rows=%zu  %s obj=%.4f  in %.3fs\n",
              NS, A.size(), s, r.objective, dt);
  if (dt > 5.0) { std::puts("  stopping scan (>5s)"); break; }
  }
  return 0;
}
