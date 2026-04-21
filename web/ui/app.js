// Frontend logic: form -> worker -> plot.
//
// The solver runs in a dedicated Web Worker (solver_worker.js) so UI stays
// responsive.  If solver.js is not yet built (e.g., the user is just
// browsing the gallery), we gracefully report this and the form shows the
// preset specifications without running anything.

const els = {
  form:     document.getElementById("spec-form"),
  run:      document.getElementById("run-btn"),
  status:   document.getElementById("status"),
  plot:     document.getElementById("plot"),
  data:     document.getElementById("filter-data"),
  nF:       document.getElementById("nF"),
  nP:       document.getElementById("nP"),
  pb:       document.getElementById("passbands"),
  sb:       document.getElementById("stopbands"),
  psi_I:    document.getElementById("psi_I_db"),
  psi_Jdef: document.getElementById("psi_J_default_db"),
  psi_Jp:   document.getElementById("psi_J_pieces"),
  base:     document.getElementById("base_samples"),
  rescale:  document.getElementById("rescale"),
};

// Preset loading
document.querySelectorAll(".load-preset").forEach((btn) => {
  btn.addEventListener("click", () => {
    const p = window.ZOLO_PRESETS[btn.dataset.preset];
    if (!p) return;
    els.nF.value = p.nF;
    els.nP.value = p.nP;
    els.pb.value = p.passbands;
    els.sb.value = p.stopbands;
    els.psi_I.value = p.psi_I_db;
    els.psi_Jdef.value = p.psi_J_default_db;
    els.psi_Jp.value = p.psi_J_pieces ?? "";
    document.getElementById("spec-form").scrollIntoView({ behavior: "smooth" });
  });
});

// Parse helpers
function parsePairs(txt) {
  return txt
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean)
    .map((l) => {
      const parts = l.split(",").map((x) => parseFloat(x));
      if (parts.length !== 2 || parts.some(Number.isNaN))
        throw new Error("bad interval line: " + l);
      return parts;
    });
}

function parsePsiPieces(txt) {
  if (!txt.trim()) return [];
  return txt
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean)
    .map((l) => {
      const parts = l.split(",").map((x) => parseFloat(x));
      if (parts.length !== 3 || parts.some(Number.isNaN))
        throw new Error("bad psi piece: " + l);
      return [[parts[0], parts[1]], parts[2]];
    });
}

function buildSpec() {
  const passbands = parsePairs(els.pb.value);
  const stopbands = parsePairs(els.sb.value);
  const omega = linspace(
    Math.min(...passbands.flat(), ...stopbands.flat()) * 1.05,
    Math.max(...passbands.flat(), ...stopbands.flat()) * 1.05,
    2000
  );
  return {
    passbands,
    stopbands,
    nF: parseInt(els.nF.value, 10),
    nP: parseInt(els.nP.value, 10),
    psi_I_db: parseFloat(els.psi_I.value),
    psi_J_default_db: parseFloat(els.psi_Jdef.value),
    psi_J_pieces: parsePsiPieces(els.psi_Jp.value),
    base_samples: parseInt(els.base.value, 10),
    rescale: els.rescale.checked,
    evaluation_omega: omega,
  };
}

function linspace(a, b, n) {
  const out = new Array(n);
  for (let i = 0; i < n; i++) out[i] = a + ((b - a) * i) / (n - 1);
  return out;
}

// Worker setup
let worker = null;
let workerAvailable = null; // tri-state: null/true/false

async function probeWorker() {
  if (workerAvailable !== null) return workerAvailable;
  try {
    const resp = await fetch("solver.js", { method: "HEAD" });
    workerAvailable = resp.ok;
  } catch (_e) {
    workerAvailable = false;
  }
  return workerAvailable;
}

function getWorker() {
  if (worker) return worker;
  worker = new Worker("solver_worker.js");
  return worker;
}

function runSolver(spec) {
  return new Promise((resolve, reject) => {
    const w = getWorker();
    w.onmessage = (ev) => {
      if (ev.data.ok) resolve(ev.data.result);
      else reject(new Error(ev.data.error));
    };
    w.onerror = (e) => reject(new Error(String(e.message ?? e)));
    w.postMessage(spec);
  });
}

function renderPlot(result, spec) {
  const r = result.response;
  if (!r) return;
  const traces = [
    { x: r.omega, y: r.S21_dB, type: "scatter", mode: "lines", name: "|S21|", line: { color: "#1b5e9b", width: 1.2 } },
    { x: r.omega, y: r.S11_dB, type: "scatter", mode: "lines", name: "|S11|", line: { color: "#b23a48", width: 1.2 } },
  ];
  const shapes = spec.stopbands.map(([a, b]) => ({
    type: "rect", xref: "x", yref: "paper", x0: a, x1: b, y0: 0, y1: 1,
    fillcolor: "rgba(180,180,180,0.25)", line: { width: 0 },
  }));
  Plotly.react(els.plot, traces, {
    title: `Certified optimum: nF=${spec.nF}, nP=${spec.nP}, M=${result.M.toFixed(3)}`,
    xaxis: { title: "ω (normalised)" },
    yaxis: { title: "dB", range: [-80, 2] },
    shapes,
    margin: { t: 40, b: 45, l: 55, r: 12 },
  }, { displayModeBar: false, responsive: true });

  els.data.textContent = JSON.stringify(
    {
      M: result.M,
      scale: result.scale,
      sigma_I: result.sigma_I,
      sigma_J: result.sigma_J,
      F_mono: result.F_mono,
      P_mono: result.P_mono,
    }, null, 2);
}

els.form.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  els.run.disabled = true;
  els.status.textContent = "solving…";

  const have = await probeWorker();
  if (!have) {
    els.status.textContent =
      "solver.wasm not found — run `emcmake cmake` build (see README).";
    els.run.disabled = false;
    return;
  }

  let spec;
  try { spec = buildSpec(); }
  catch (e) { els.status.textContent = e.message; els.run.disabled = false; return; }

  const started = performance.now();
  try {
    const result = await runSolver(spec);
    const dt = (performance.now() - started) / 1000;
    if (!result.success) {
      els.status.textContent = "failed: " + (result.message ?? "unknown");
    } else {
      els.status.textContent = `done in ${dt.toFixed(2)} s`;
      renderPlot(result, spec);
    }
  } catch (e) {
    els.status.textContent = "error: " + e.message;
  } finally {
    els.run.disabled = false;
  }
});

// Nice-to-have: on page load, if WASM isn't built yet, hint about it.
probeWorker().then((ok) => {
  if (!ok) {
    els.status.textContent =
      "solver.wasm not yet built — use the gallery, or build WASM per README.";
  }
});
