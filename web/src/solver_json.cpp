// JSON marshalling and Emscripten bindings for the Zolotarev solver.
//
// JSON layout:
//
//   input  = {
//     "passbands":  [[a, b], ...],
//     "stopbands":  [[a, b], ...],
//     "nF": int, "nP": int,
//     "psi_I_db":  float,              # uniform passband return loss in dB
//     "psi_J_default_db": float,       # default stopband rejection in dB
//     "psi_J_pieces": [ [[a, b], db], ...],  # optional per-range overrides
//     "evaluation_omega": [w1, w2, ...]      # optional response sample grid
//   }
//
//   output = {
//     "success": bool, "message": str,
//     "F_mono": [...],  "P_mono": [...],
//     "M": float, "scale": float,
//     "sigma_I": [...], "sigma_J": [...],
//     "response": {"omega":[...], "S21_dB":[...], "S11_dB":[...]}
//   }
#include "solver.hpp"
#include "mini_json.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#ifdef __EMSCRIPTEN__
#include <emscripten/bind.h>
#endif

namespace zolo {
namespace {

double db_of_abs_sq(double abs_sq) {
  double x = std::max(abs_sq, 1e-40);
  return 10.0 * std::log10(x);
}

mjson::Array to_array(const std::vector<double>& v) {
  mjson::Array a;
  a.reserve(v.size());
  for (double x : v) a.emplace_back(x);
  return a;
}
mjson::Array to_array(const std::vector<int>& v) {
  mjson::Array a;
  a.reserve(v.size());
  for (int x : v) a.emplace_back(static_cast<double>(x));
  return a;
}

}  // namespace

std::string solve_json(const std::string& spec_json) {
  mjson::Value out;
  try {
    mjson::Value in = mjson::Value::parse(spec_json);

    Spec spec;
    for (const auto& ab : in.at("passbands").as_arr()) {
      spec.passbands.push_back({ab.as_arr()[0].as_num(),
                                ab.as_arr()[1].as_num()});
    }
    for (const auto& ab : in.at("stopbands").as_arr()) {
      spec.stopbands.push_back({ab.as_arr()[0].as_num(),
                                ab.as_arr()[1].as_num()});
    }
    spec.nF = in.at("nF").as_int();
    spec.nP = in.at("nP").as_int();

    // psi_I_linear takes precedence over psi_I_db; rl_db_to_psi(0) is
    // infinite (no return-loss constraint) which isn't a useful default,
    // so we allow callers to specify the linear value directly.
    double psi_I_const;
    if (in.contains("psi_I_linear")) {
      psi_I_const = in.at("psi_I_linear").as_num();
    } else {
      psi_I_const = rl_db_to_psi(in.value_num("psi_I_db", 20.0));
    }
    double psi_J_default = rej_db_to_psi(in.value_num("psi_J_default_db", 0.0));

    struct Piece { double a; double b; double psi; };
    std::vector<Piece> pieces;
    if (in.contains("psi_J_pieces")) {
      for (const auto& e : in.at("psi_J_pieces").as_arr()) {
        const auto& ab = e.as_arr()[0].as_arr();
        pieces.push_back({ab[0].as_num(), ab[1].as_num(),
                          rej_db_to_psi(e.as_arr()[1].as_num())});
      }
    }

    spec.psi_I = [psi_I_const](double) { return psi_I_const; };
    spec.psi_J = [psi_J_default, pieces](double w) {
      for (const auto& p : pieces) {
        if (p.a <= w && w <= p.b) return p.psi;
      }
      return psi_J_default;
    };

    if (in.contains("trace")) set_trace(in.at("trace").as_bool());
    else set_trace(false);
    spec.base_samples   = in.value_int("base_samples",   spec.base_samples);
    spec.refine_samples = in.value_int("refine_samples", spec.refine_samples);
    spec.max_iter       = in.value_int("max_iter",       spec.max_iter);
    spec.coef_bound     = in.value_num("coef_bound",     spec.coef_bound);
    spec.tol            = in.value_num("tol",            spec.tol);
    spec.rescale        = in.value_bool("rescale",       spec.rescale);

    Result r = solve_zolotarev(spec);

    out["success"] = mjson::Value(r.success);
    out["message"] = mjson::Value(r.message);
    if (r.success) {
      out["F_mono"] = mjson::Value(to_array(r.F_mono));
      out["P_mono"] = mjson::Value(to_array(r.P_mono));
      out["M"]      = mjson::Value(r.M);
      out["scale"]  = mjson::Value(r.scale);
      out["sigma_I"] = mjson::Value(to_array(r.sigma_I));
      out["sigma_J"] = mjson::Value(to_array(r.sigma_J));

      if (in.contains("evaluation_omega")) {
        std::vector<double> w;
        for (const auto& x : in.at("evaluation_omega").as_arr()) {
          w.push_back(x.as_num());
        }
        Response resp = scattering_response(r.F_mono, r.P_mono, w);
        std::vector<double> s21(resp.omega.size()), s11(resp.omega.size());
        for (size_t i = 0; i < resp.omega.size(); ++i) {
          s21[i] = db_of_abs_sq(resp.S21_sq[i]);
          s11[i] = db_of_abs_sq(resp.S11_sq[i]);
        }
        mjson::Value rj;
        rj["omega"]  = mjson::Value(to_array(resp.omega));
        rj["S21_dB"] = mjson::Value(to_array(s21));
        rj["S11_dB"] = mjson::Value(to_array(s11));
        out["response"] = rj;
      }
    }
  } catch (const std::exception& e) {
    out["success"] = mjson::Value(false);
    out["message"] = mjson::Value(std::string("exception: ") + e.what());
  }
  return out.dump();
}

}  // namespace zolo


#ifdef __EMSCRIPTEN__
EMSCRIPTEN_BINDINGS(zolo_module) {
  emscripten::function("solve_json", &zolo::solve_json);
}
#endif
