// Native smoke test: invokes the C++ solver on the Chebyshev T_2 toy case
// and prints results.  Used to verify the port against the Python
// reference before we compile to WebAssembly.
#include "solver.hpp"

#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>

int main() {
  using namespace zolo;

  // Sanity: reproduce Chebyshev T_2 with a single passband and stopband.
  {
    Spec s;
    s.passbands = {{-1.0, 1.0}};
    s.stopbands = {{1.1, 2.0}};
    s.nF = 2; s.nP = 0;
    s.psi_I = [](double) { return 1.0; };
    s.psi_J = [](double) { return 0.0; };
    s.base_samples = 40;

    Result r = solve_zolotarev(s);
    std::printf("Chebyshev T_2 test:  M = %.4f  (expected 1.4200)\n", r.M);
    std::printf("  F_mono = [");
    for (double c : r.F_mono) std::printf("%g, ", c);
    std::printf("]   (expected [-1, 0, 2])\n");
  }

  // Multiband 6-2 smoke
  {
    Spec s;
    s.passbands = {{-1.0, -0.5}, {0.5, 1.0}};
    s.stopbands = {{-2.0, -1.2}, {-0.3, 0.3}, {1.2, 2.0}};
    s.nF = 6; s.nP = 2;
    s.psi_I = [](double) { return rl_db_to_psi(20.0); };
    s.psi_J = [](double) { return 0.0; };
    s.base_samples = 40;
    Result r = solve_zolotarev(s);
    std::printf("\n6-2 dual-band:  M = %.4f   (rejection = %.2f dB)\n",
                r.M, 10.0 * std::log10(1.0 + r.M * r.M));
    std::printf("  success = %d, message = %s\n", r.success ? 1 : 0,
                r.message.c_str());
  }

  // JSON interface round-trip
  {
    std::string spec_json = R"({
      "passbands": [[-1.0, 1.0]],
      "stopbands": [[1.1, 2.0]],
      "nF": 2, "nP": 0,
      "psi_I_db": 0.0, "psi_J_default_db": 0.0,
      "evaluation_omega": [-1.0, 0.0, 1.0, 1.1, 2.0]
    })";
    std::string out = solve_json(spec_json);
    std::printf("\nJSON round-trip (truncated):\n%s\n",
                out.substr(0, std::min<size_t>(out.size(), 400)).c_str());
  }
  return 0;
}
