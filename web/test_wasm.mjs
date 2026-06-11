// Node test harness: exercise web/ui/solver.{js,wasm} exactly the way
// the browser would, but headless.  Compares results to the Python
// reference (numbers hard-coded below from examples.py runs).
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { readFileSync } from "node:fs";
import createSolverModule from "./ui/solver.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
// The WASM was built with ENVIRONMENT=web,worker (what the browser needs).
// For headless Node testing, point the loader at the local wasm file.
const wasmPath = join(__dirname, "ui", "solver.wasm");
const wasmBinary = readFileSync(wasmPath);

function dbOfD(D) {
  return 10 * Math.log10(1 + D * D);
}

function linspace(a, b, n) {
  const out = new Array(n);
  for (let i = 0; i < n; i++) out[i] = a + ((b - a) * i) / (n - 1);
  return out;
}

function polyval(coeffs, x) {
  let a = 0;
  for (let i = coeffs.length - 1; i >= 0; i--) a = a * x + coeffs[i];
  return a;
}

function analyseBands(F, P, bands, kind) {
  for (const [a, b] of bands) {
    const xs = linspace(a, b, 20000);
    let min = +Infinity, max = -Infinity;
    for (const x of xs) {
      const r = Math.abs(polyval(F, x) / polyval(P, x));
      if (r < min) min = r;
      if (r > max) max = r;
    }
    const v = kind === "I" ? max : min;
    const dB = kind === "I"
      ? -20 * Math.log10(v / Math.sqrt(1 + v * v))
      : dbOfD(v);
    console.log(
      `    ${kind}=[${a.toFixed(3)}, ${b.toFixed(3)}]` +
      `  ${kind === "I" ? "max" : "min"}|F/P|=${v.toFixed(4)}` +
      `  ${kind === "I" ? "RL" : "rej"}=${dB.toFixed(2)} dB`
    );
  }
}

// Dense passband-validity check: the returned filter must honour the
// return-loss spec on the continuum (a candidate that cheats between LP
// samples is a solver bug, whatever its M claims).
function passbandValid(F, P, passbands, psiI, slackDb = 0.15) {
  for (const [a, b] of passbands) {
    const xs = linspace(a, b, 20000);
    let max = 0;
    for (const x of xs) {
      const r = Math.abs(polyval(F, x) / polyval(P, x));
      if (r > max) max = r;
    }
    const rlAch = -20 * Math.log10(max / Math.sqrt(1 + max * max));
    const rlTarget = -20 * Math.log10(psiI / Math.sqrt(1 + psiI * psiI));
    if (rlAch < rlTarget - slackDb) {
      console.log(
        `  PASSBAND VIOLATION on [${a}, ${b}]: ` +
        `achieved RL ${rlAch.toFixed(2)} dB < target ${rlTarget.toFixed(2)} dB`
      );
      return false;
    }
  }
  return true;
}

function runCase(Module, name, spec, expected) {
  console.log(`\n== ${name} ==`);
  const t0 = performance.now();
  const resJson = Module.solve_json(JSON.stringify(spec));
  const dt = (performance.now() - t0) / 1000;
  const res = JSON.parse(resJson);
  if (!res.success) {
    console.log(`  FAILED: ${res.message}`);
    return false;
  }
  const ub = res.M_upper !== undefined ? res.M_upper.toFixed(4) : "?";
  const alt = res.alternation
    ? `${res.alternation.count}/${res.alternation.required}` +
      (res.alternation.certified ? " certified" : "")
    : "?";
  console.log(
    `  took ${dt.toFixed(2)} s   M=${res.M.toFixed(4)} (<= ${ub})   ` +
    `alt=${alt}   sigma_I=[${res.sigma_I}]   sigma_J=[${res.sigma_J}]`
  );
  analyseBands(res.F_mono, res.P_mono, spec.passbands, "I");
  analyseBands(res.F_mono, res.P_mono, spec.stopbands, "J");

  let ok = true;
  // Certificate sanity: the bracket must contain the achieved M and the
  // alternation block must be reported.
  if (res.M_upper !== undefined && res.M_upper < res.M - 1e-9) {
    console.log(`  BAD BRACKET: M_upper=${res.M_upper} < M=${res.M}`);
    ok = false;
  }
  if (!res.alternation) {
    console.log("  MISSING alternation certificate block");
    ok = false;
  }
  const psiI = spec.psi_I_linear ??
    (10 ** (-spec.psi_I_db / 20)) / Math.sqrt(1 - 10 ** (-spec.psi_I_db / 10));
  if (!passbandValid(res.F_mono, res.P_mono, spec.passbands, psiI)) ok = false;

  if (expected) {
    const tolM = expected.tol ?? 0.4;
    const okM = Math.abs(res.M - expected.M) < tolM;
    console.log(
      `  expected M~=${expected.M.toFixed(3)}  ` +
      `(delta=${(res.M - expected.M).toFixed(3)})  ` +
      (okM ? "OK" : "MISMATCH")
    );
    ok = ok && okM;
  }
  return ok;
}

const Module = await createSolverModule({ wasmBinary });

let pass = 0, fail = 0;

// Sanity: Chebyshev T_2 on a single band.  The classical Zolotarev uses
// |F/P| <= 1 on I, so pass psi_I_linear=1 directly (psi_I_db=0 would mean
// infinite psi -- no RL constraint).  Expected T_2(1.1) = 1.42.
if (runCase(Module, "Chebyshev T_2 single-band",
  {
    passbands: [[-1, 1]],
    stopbands: [[1.1, 2]],
    nF: 2, nP: 0,
    psi_I_linear: 1.0, psi_J_default_db: 0,
    base_samples: 30,
  },
  { M: 1.42, tol: 0.02 })) pass++; else fail++;

// Example 1 geometry, uniform psi_J=0.  Reference: M~6.95 (Python and
// C++ agree to 4 decimals after the feasibility-kick refinement; the
// previously documented 2.39 was a stuck local optimum with a false
// infeasibility certificate).
if (runCase(Module, "Example 1 geometry, uniform psi_J=0",
  {
    passbands: [[-1, -0.625], [0.25, 1]],
    stopbands: [[-10, -1.188], [-0.5, 0.125], [1.212, 10]],
    nF: 9, nP: 3,
    psi_I_db: 20, psi_J_default_db: 0,
    base_samples: 30,
  },
  { M: 6.95, tol: 0.4 })) pass++; else fail++;

// Example 2 geometry, uniform psi_J=0.  Reference: M~15.84 (was 15.16
// before the refinement; 10.22 and 7.29 were earlier stuck optima).
if (runCase(Module, "Example 2 geometry, uniform psi_J=0",
  {
    passbands: [[-1, -0.383], [0.383, 1]],
    stopbands: [[-10, -1.864], [-0.037, -0.012], [1.185, 10]],
    nF: 7, nP: 3,
    psi_I_db: 23, psi_J_default_db: 0,
    base_samples: 30,
  },
  { M: 15.84, tol: 0.5 })) pass++; else fail++;

// Example 1 ASYMMETRIC (paper's 15/30 dB spec).  M is the paper's
// multiplicatively weighted criterion min |F/P| / psi_J: native C++
// reference 0.3658 (i.e. the spec is uniformly missed by ~8.7 dB --
// the spec is intentionally infeasible for a 9-3).  The historical
// -30.80 came from an additive misreading of the spec mechanism whose
// negative-Meff rows silently over-constrained the outer bands.
if (runCase(Module, "Example 1 asymmetric 15/30 dB",
  {
    passbands: [[-1, -0.625], [0.25, 1]],
    stopbands: [[-10, -1.188], [-0.5, 0.125], [1.212, 10]],
    nF: 9, nP: 3,
    psi_I_db: 20, psi_J_default_db: 15,
    psi_J_pieces: [[[-0.5, 0.125], 30]],
    base_samples: 30,
  },
  { M: 0.366, tol: 0.12 })) pass++; else fail++;

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail === 0 ? 0 : 1);
