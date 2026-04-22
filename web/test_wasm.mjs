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
  console.log(
    `  took ${dt.toFixed(2)} s   M=${res.M.toFixed(4)}   ` +
    `sigma_I=[${res.sigma_I}]   sigma_J=[${res.sigma_J}]`
  );
  analyseBands(res.F_mono, res.P_mono, spec.passbands, "I");
  analyseBands(res.F_mono, res.P_mono, spec.stopbands, "J");

  if (expected) {
    const tolM = expected.tol ?? 0.4;
    const ok = Math.abs(res.M - expected.M) < tolM;
    console.log(
      `  expected M~=${expected.M.toFixed(3)}  ` +
      `(delta=${(res.M - expected.M).toFixed(3)})  ` +
      (ok ? "OK" : "MISMATCH")
    );
    return ok;
  }
  return true;
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
    base_samples: 40,
  },
  { M: 1.42 })) pass++; else fail++;

// Example 1 geometry, uniform psi_J=0.  Python reference: M=2.39.
if (runCase(Module, "Example 1 geometry, uniform psi_J=0",
  {
    passbands: [[-1, -0.625], [0.25, 1]],
    stopbands: [[-10, -1.188], [-0.5, 0.125], [1.212, 10]],
    nF: 9, nP: 3,
    psi_I_db: 20, psi_J_default_db: 0,
    base_samples: 40,
  },
  { M: 2.39 })) pass++; else fail++;

// Example 2 geometry, uniform psi_J=0.  Python reference: M=10.22.
if (runCase(Module, "Example 2 geometry, uniform psi_J=0",
  {
    passbands: [[-1, -0.383], [0.383, 1]],
    stopbands: [[-10, -1.864], [-0.037, -0.012], [1.185, 10]],
    nF: 7, nP: 3,
    psi_I_db: 23, psi_J_default_db: 0,
    base_samples: 40,
  },
  { M: 10.22 })) pass++; else fail++;

// Example 1 ASYMMETRIC (paper's 15/30 dB spec).  Python reference: M=-30.80.
if (runCase(Module, "Example 1 asymmetric 15/30 dB",
  {
    passbands: [[-1, -0.625], [0.25, 1]],
    stopbands: [[-10, -1.188], [-0.5, 0.125], [1.212, 10]],
    nF: 9, nP: 3,
    psi_I_db: 20, psi_J_default_db: 15,
    psi_J_pieces: [[[-0.5, 0.125], 30]],
    base_samples: 40,
  },
  { M: -30.80 })) pass++; else fail++;

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail === 0 ? 0 : 1);
