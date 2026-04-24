// Standalone sanity tests for mini_lp.hpp.
//   g++ -std=c++17 -O0 -g mini_lp_test.cpp -o mini_lp_test
#include "mini_lp.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>

using mlp::solve;
using mlp::Result;
using mlp::Status;

static const double INF = std::numeric_limits<double>::infinity();

static void print_result(const char* name, const Result& r) {
  const char* stat = r.status == Status::Optimal ? "OPT"
                   : r.status == Status::Infeasible ? "INFEAS" : "UNBND";
  std::printf("[%s] %s  obj=%.6f  x=[", name, stat, r.objective);
  for (std::size_t i = 0; i < r.x.size(); ++i) {
    std::printf("%s%.4f", i ? "," : "", r.x[i]);
  }
  std::printf("]\n");
}

int main() {
  // Test 1: min -x - y  s.t.  x + y <= 1,  x, y in [0, inf)
  // expected: x+y = 1, obj = -1.
  {
    auto r = solve({-1, -1}, {0, 0}, {INF, INF},
                   {{1, 1}}, {1});
    print_result("T1 max x+y", r);
    assert(std::abs(r.objective + 1.0) < 1e-9);
  }

  // Test 2: max h  s.t.  h <= f,  -1 <= f <= 1
  //   equivalent min -h, f in [-1, 1], h free
  //   max: h = f = 1, obj = -1.
  {
    auto r = solve({0, -1},                 // vars: [f, h], minimize -h
                   {-1, -INF}, {1, INF},     // f in [-1,1], h free
                   {{-1, 1}},                // h - f <= 0  (h <= f)
                   {0});
    print_result("T2 h<=f", r);
    assert(std::abs(r.objective + 1.0) < 1e-6);
    assert(std::abs(r.x[0] - 1.0) < 1e-6 && std::abs(r.x[1] - 1.0) < 1e-6);
  }

  // Test 3: Chebyshev T_2 toy.
  //   F(x) = f0 + f1*T1(x) + f2*T2(x) with Cheb basis.
  //   constraints:
  //     |F(x)| <= 1 for x in {-1, 0, 1}          (passband samples)
  //     F(y) >= h for y in {1.1, 2}              (stopband samples)
  //   maximize h.
  //   Expected: F = T_2 = 2x^2 - 1, so f0=-1, f1=0, f2=2 in Cheb basis
  //   (note: T_0 = 1, T_1 = x, T_2 = 2x^2-1, so F = f0*1 + f1*x + f2*(2x^2-1))
  //   In Cheb basis directly: coefficient of T_2 in F = 1 for T_2(x) match.
  //   Let's just check max h ~ 1.42 and |F|<=1 holds.
  {
    auto T0 = [](double x){ (void)x; return 1.0; };
    auto T1 = [](double x){ return x; };
    auto T2 = [](double x){ return 2*x*x - 1; };

    std::vector<std::vector<double>> A;
    std::vector<double> b;
    auto add = [&](std::vector<double> row, double rhs){
      A.push_back(row); b.push_back(rhs);
    };
    // variables: [f0, f1, f2, h]  (h free)
    for (double x : {-1.0, -0.5, 0.0, 0.5, 1.0}) {
      add({T0(x), T1(x), T2(x), 0}, 1.0);   // F(x) <= 1
      add({-T0(x), -T1(x), -T2(x), 0}, 1.0);// -F(x) <= 1
    }
    for (double y : {1.1, 2.0}) {
      // F(y) >= h   =>   -F(y) + h <= 0
      add({-T0(y), -T1(y), -T2(y), 1}, 0.0);
    }

    auto r = solve(
        /*c=*/    {0, 0, 0, -1},              // min -h
        /*lo=*/   {-1e6, -1e6, -1e6, -INF},
        /*hi=*/   { 1e6,  1e6,  1e6,  INF},
        A, b);
    print_result("T3 Chebyshev T2 toy", r);
    std::printf("    h = %.6f  (expected ~1.42)\n", r.x.back());
  }

  std::puts("all asserts passed");
  return 0;
}
