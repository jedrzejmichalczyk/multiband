// Frontend logic for the multiband Zolotarev UI.
// Band spec is built from the table rows, a Web Worker runs the WASM
// solver, and we plot |S21|, |S11|, and the user's targets on Plotly.

const $ = (s) => document.querySelector(s);

const els = {
  nF:       $("#nF"),
  nP:       $("#nP"),
  passRows: $("#pass-rows"),
  stopRows: $("#stop-rows"),
  runBtn:   $("#run-btn"),
  status:   $("#status"),
  results:  $("#results"),
  mBadge:   $("#m-badge"),
  plot:     $("#plot"),
  tblPass:  $("#tbl-pass"),
  tblStop:  $("#tbl-stop"),
  zF:       $("#zeros-F"),
  zP:       $("#zeros-P"),
  coeffs:   $("#coeffs"),
};

// ----- Band table helpers ---------------------------------------------------

function bandRowHTML(kind, v) {
  const psiLabel = kind === "pass" ? "rl_db" : "rej_db";
  return `
    <td><input class="w-a" type="number" step="0.001" value="${v.a}"></td>
    <td><input class="w-b" type="number" step="0.001" value="${v.b}"></td>
    <td><input class="w-psi" type="number" step="0.5"
               value="${v[psiLabel]}"></td>
    <td><button type="button" class="row-del" title="remove">×</button></td>`;
}

function addBandRow(kind, v) {
  const tbody = kind === "pass" ? els.passRows : els.stopRows;
  const tr = document.createElement("tr");
  tr.innerHTML = bandRowHTML(kind, v);
  tr.querySelector(".row-del").addEventListener("click", () => tr.remove());
  tbody.appendChild(tr);
}

function clearBands() {
  els.passRows.innerHTML = "";
  els.stopRows.innerHTML = "";
}

function readBandRows(tbody, psiKey) {
  const out = [];
  for (const tr of tbody.children) {
    const a = parseFloat(tr.querySelector(".w-a").value);
    const b = parseFloat(tr.querySelector(".w-b").value);
    const psi = parseFloat(tr.querySelector(".w-psi").value);
    if (!Number.isFinite(a) || !Number.isFinite(b) || !Number.isFinite(psi))
      throw new Error("invalid number in band row");
    if (b <= a) throw new Error(`band [${a}, ${b}] is empty`);
    const row = { a, b }; row[psiKey] = psi; out.push(row);
  }
  return out;
}

// ----- Preset loading -------------------------------------------------------

function applyPreset(key, { run = true } = {}) {
  const p = window.ZOLO_PRESETS[key];
  if (!p) return;
  els.nF.value = p.nF;
  els.nP.value = p.nP;
  clearBands();
  p.passbands.forEach((b) => addBandRow("pass", b));
  p.stopbands.forEach((b) => addBandRow("stop", b));
  if (run) runSolver();
}

document.querySelectorAll(".preset").forEach((btn) => {
  btn.addEventListener("click", () => applyPreset(btn.dataset.preset));
});
document.querySelectorAll(".row-add").forEach((btn) => {
  btn.addEventListener("click", () => {
    const kind = btn.dataset.kind;
    const last = kind === "pass" ? els.passRows.lastElementChild : els.stopRows.lastElementChild;
    const psiKey = kind === "pass" ? "rl_db" : "rej_db";
    const base = last
      ? { a: parseFloat(last.querySelector(".w-b").value),
          b: parseFloat(last.querySelector(".w-b").value) + 0.2,
          [psiKey]: parseFloat(last.querySelector(".w-psi").value) }
      : { a: 0, b: 1, [psiKey]: kind === "pass" ? 20 : 15 };
    addBandRow(kind, base);
  });
});

// Populate the form with a sensible default on first load, but don't run
// the solver until the user clicks Synthesise or a preset button.
applyPreset("ex1_uniform", { run: false });

// ----- Build the JSON spec for the solver -----------------------------------

function linspace(a, b, n) {
  const out = new Array(n);
  for (let i = 0; i < n; i++) out[i] = a + ((b - a) * i) / (n - 1);
  return out;
}

function buildSpec() {
  const passbands = readBandRows(els.passRows, "rl_db");
  const stopbands = readBandRows(els.stopRows, "rej_db");
  if (!passbands.length) throw new Error("need at least one passband");
  if (!stopbands.length) throw new Error("need at least one stopband");
  const nF = parseInt(els.nF.value, 10);
  const nP = parseInt(els.nP.value, 10);

  // All passbands currently share one return-loss target (tightest wins).
  const psi_I_db = Math.max(...passbands.map((p) => p.rl_db));

  // Per-interval stopband targets.  We reuse the "pieces" mechanism for
  // this.  The default outside pieces is 0 dB (no bias).
  const psi_J_pieces = stopbands.map((s) => [[s.a, s.b], s.rej_db]);

  const allEdges = [...passbands.flatMap((p) => [p.a, p.b]),
                    ...stopbands.flatMap((s) => [s.a, s.b])];
  const lo = Math.min(...allEdges) * 1.05;
  const hi = Math.max(...allEdges) * 1.05;

  return {
    spec: {
      passbands: passbands.map((p) => [p.a, p.b]),
      stopbands: stopbands.map((s) => [s.a, s.b]),
      nF, nP,
      psi_I_db,
      psi_J_default_db: 0,
      psi_J_pieces,
      base_samples: 40,
      evaluation_omega: linspace(lo, hi, 2400),
    },
    passbands,
    stopbands,
    plotRange: [lo, hi],
    nF, nP,
  };
}

// ----- Worker wrapping ------------------------------------------------------

let worker = null;
function getWorker() {
  if (!worker) worker = new Worker("solver_worker.js");
  return worker;
}

function solve(spec) {
  return new Promise((resolve, reject) => {
    const w = getWorker();
    w.onmessage = (ev) => ev.data.ok ? resolve(ev.data.result) : reject(new Error(ev.data.error));
    w.onerror   = (e)  => reject(new Error(String(e.message ?? e)));
    w.postMessage(spec);
  });
}

// ----- Polynomial evaluation (for per-band analysis) ------------------------

function polyval(c, x) {
  let a = 0;
  for (let i = c.length - 1; i >= 0; i--) a = a * x + c[i];
  return a;
}

function minAbsDOverInterval(F, P, a, b, n = 4000) {
  let min = +Infinity, max = -Infinity;
  for (let i = 0; i < n; i++) {
    const x = a + ((b - a) * i) / (n - 1);
    const r = Math.abs(polyval(F, x) / polyval(P, x));
    if (r < min) min = r;
    if (r > max) max = r;
  }
  return { min, max };
}

// Poly roots: use companion-matrix approach via Aberth iteration would be
// nice, but the polynomial is small.  We use Durand-Kerner (all roots at
// once).  Coefficients ascending.  Returns real parts only — for our
// filters the optimum is real-coefficient with roots predominantly real;
// complex conjugate pairs become a single magnitude on the axis.
function polyRoots(c) {
  const n = c.length - 1;
  if (n < 1) return [];
  // Normalise: leading coefficient to 1.
  const lead = c[n];
  if (!Number.isFinite(lead) || Math.abs(lead) < 1e-14) return [];
  const a = c.map((v) => v / lead);
  // Durand-Kerner with initial points around unit circle.
  let roots = [];
  for (let k = 0; k < n; k++) {
    const th = (2 * Math.PI * k) / n;
    roots.push({ re: 0.4 * Math.cos(th), im: 0.4 * Math.sin(th) });
  }
  const horner = (z) => {
    let re = 1, im = 0;
    for (let i = n - 1; i >= 0; i--) {
      const nre = re * z.re - im * z.im + a[i];
      const nim = re * z.im + im * z.re;
      re = nre; im = nim;
    }
    return { re, im };
  };
  for (let it = 0; it < 200; it++) {
    let maxDelta = 0;
    const next = roots.map((ri, i) => {
      let dre = 1, dim = 0;
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        const dr = ri.re - roots[j].re;
        const di = ri.im - roots[j].im;
        const nre = dre * dr - dim * di;
        const nim = dre * di + dim * dr;
        dre = nre; dim = nim;
      }
      const h = horner(ri);
      // delta = horner(ri) / prod(ri - rj)
      const denom = dre * dre + dim * dim;
      const dxre = (h.re * dre + h.im * dim) / denom;
      const dxim = (h.im * dre - h.re * dim) / denom;
      const nr = { re: ri.re - dxre, im: ri.im - dxim };
      const md = Math.hypot(dxre, dxim);
      if (md > maxDelta) maxDelta = md;
      return nr;
    });
    roots = next;
    if (maxDelta < 1e-12) break;
  }
  return roots.sort((a, b) => a.re - b.re);
}

function formatRoot(r) {
  const re = r.re.toFixed(4);
  if (Math.abs(r.im) < 1e-5) return re;
  return `${re} ${r.im > 0 ? "+" : "−"} ${Math.abs(r.im).toFixed(4)} i`;
}

// ----- Rendering ------------------------------------------------------------

function psiRL(rl_db) {
  const r = Math.pow(10, -rl_db / 20);
  return r / Math.sqrt(1 - r * r);
}
function psiRej(rej_db) {
  return Math.sqrt(Math.pow(10, rej_db / 10) - 1);
}
function rejDBFromD(D) {
  return 10 * Math.log10(1 + D * D);
}

function renderPlot(result, ctx) {
  const r = result.response;
  if (!r || !window.Plotly) return;

  const traces = [
    { x: r.omega, y: r.S21_dB, type: "scatter", mode: "lines",
      name: "|S21|", line: { color: "#1b5e9b", width: 1.4 } },
    { x: r.omega, y: r.S11_dB, type: "scatter", mode: "lines",
      name: "|S11|", line: { color: "#b23a48", width: 1.4 } },
  ];

  const shapes = [];
  for (const b of ctx.passbands) {
    shapes.push({
      type: "rect", xref: "x", yref: "paper",
      x0: b.a, x1: b.b, y0: 0, y1: 1,
      fillcolor: "rgba(42,124,42,0.09)", line: { width: 0 }, layer: "below",
    });
    // Return-loss target line (|S11| = -RL dB in the passband)
    shapes.push({
      type: "line", xref: "x", yref: "y",
      x0: b.a, x1: b.b, y0: -b.rl_db, y1: -b.rl_db,
      line: { color: "#b23a48", width: 1, dash: "dash" },
    });
  }
  for (const s of ctx.stopbands) {
    shapes.push({
      type: "rect", xref: "x", yref: "paper",
      x0: s.a, x1: s.b, y0: 0, y1: 1,
      fillcolor: "rgba(184,52,45,0.07)", line: { width: 0 }, layer: "below",
    });
    if (s.rej_db > 0) {
      // Rejection target line (|S21| = -rej dB)
      shapes.push({
        type: "line", xref: "x", yref: "y",
        x0: s.a, x1: s.b, y0: -s.rej_db, y1: -s.rej_db,
        line: { color: "#1b5e9b", width: 1, dash: "dash" },
      });
    }
  }

  Plotly.react(els.plot, traces, {
    xaxis: { title: "ω (normalised)", range: ctx.plotRange, zeroline: false, gridcolor: "#eee" },
    yaxis: { title: "dB", range: [-80, 3], zeroline: false, gridcolor: "#eee" },
    shapes,
    margin: { t: 10, b: 45, l: 55, r: 12 },
    legend: { orientation: "h", x: 0, y: -0.18 },
    plot_bgcolor: "#fff", paper_bgcolor: "#fff",
  }, { displayModeBar: false, responsive: true });
}

function renderSummary(result, ctx) {
  // passbands
  let html = "<thead><tr><th>Interval</th><th>target RL</th><th>achieved RL</th></tr></thead><tbody>";
  for (const b of ctx.passbands) {
    const { max } = minAbsDOverInterval(result.F_mono, result.P_mono, b.a, b.b);
    // RL = -20 log10 |S11| = -20 log10 (D/sqrt(1+D²))
    const rlAch = max > 0
      ? -20 * Math.log10(max / Math.sqrt(1 + max * max))
      : Infinity;
    const ok = rlAch + 0.05 >= b.rl_db;  // slight tol for sampling
    html += `<tr>
      <td>[${b.a.toFixed(3)}, ${b.b.toFixed(3)}]</td>
      <td class="num">${b.rl_db.toFixed(1)} dB</td>
      <td class="num ${ok ? "ok" : "fail"}">${rlAch.toFixed(2)} dB</td>
    </tr>`;
  }
  html += "</tbody>";
  els.tblPass.innerHTML = html;

  // stopbands
  html = "<thead><tr><th>Interval</th><th>target rej</th><th>achieved rej</th></tr></thead><tbody>";
  for (const s of ctx.stopbands) {
    const { min } = minAbsDOverInterval(result.F_mono, result.P_mono, s.a, s.b);
    const rejAch = rejDBFromD(min);
    const ok = rejAch + 0.05 >= s.rej_db;
    html += `<tr>
      <td>[${s.a.toFixed(3)}, ${s.b.toFixed(3)}]</td>
      <td class="num">${s.rej_db.toFixed(1)} dB</td>
      <td class="num ${ok ? "ok" : "fail"}">${rejAch.toFixed(2)} dB</td>
    </tr>`;
  }
  html += "</tbody>";
  els.tblStop.innerHTML = html;
}

function renderZeros(result) {
  const rootsF = polyRoots(result.F_mono);
  const rootsP = polyRoots(result.P_mono);
  els.zF.innerHTML = rootsF.map((r) => `<li>${formatRoot(r)}</li>`).join("");
  els.zP.innerHTML = rootsP.length
    ? rootsP.map((r) => `<li>${formatRoot(r)}</li>`).join("")
    : "<li><em>all-pole filter (P is constant)</em></li>";
  els.coeffs.textContent =
    "F_mono = [\n  " + result.F_mono.map((x) => x.toExponential(6)).join(",\n  ") + "\n]\n\n" +
    "P_mono = [\n  " + result.P_mono.map((x) => x.toExponential(6)).join(",\n  ") + "\n]";
}

function renderMBadge(M, ctx) {
  // M = min over stopbands of (|F/P| − ψ_J).  If ≥0 every target is met.
  els.mBadge.classList.remove("ok", "fail");
  let html, cls;
  if (M >= 0) {
    html = `M = +${M.toFixed(3)} &nbsp;·&nbsp; every stopband target is met (with slack)`;
    cls = "ok";
  } else {
    html = `M = ${M.toFixed(3)} &nbsp;·&nbsp; a ${ctx.nF}-${ctx.nP} filter cannot meet every target simultaneously`;
    cls = "fail";
  }
  els.mBadge.innerHTML = html;
  els.mBadge.classList.add(cls);
}

// ----- Main ----------------------------------------------------------------

async function runSolver() {
  let built;
  try { built = buildSpec(); }
  catch (e) { els.status.textContent = e.message; els.status.className = "warn"; return; }

  els.runBtn.disabled = true;
  els.status.className = "busy";
  els.status.textContent = "solving…";

  const t0 = performance.now();
  try {
    const result = await solve(built.spec);
    const dt = (performance.now() - t0) / 1000;
    if (!result.success) {
      els.status.className = "warn";
      els.status.textContent = "failed: " + (result.message ?? "unknown");
      return;
    }
    els.results.classList.remove("hidden");
    els.status.className = "";
    els.status.textContent = `solved in ${dt.toFixed(2)} s  (σ_I=[${result.sigma_I}], σ_J=[${result.sigma_J}])`;

    renderMBadge(result.M, built);
    renderPlot(result, built);
    renderSummary(result, built);
    renderZeros(result);
  } catch (e) {
    els.status.className = "warn";
    els.status.textContent = "error: " + e.message;
  } finally {
    els.runBtn.disabled = false;
  }
}

els.runBtn.addEventListener("click", runSolver);

// Probe whether the WASM artefacts exist (yellow warning on first load).
fetch("solver.js", { method: "HEAD" }).then((r) => {
  if (!r.ok) {
    els.status.className = "warn";
    els.status.textContent =
      "solver.wasm not yet built — see web/README.md for the emsdk build steps.";
    els.runBtn.disabled = true;
  }
}).catch(() => {
  els.status.className = "warn";
  els.status.textContent = "solver.wasm not yet built — serve this directory with a local HTTP server.";
});
