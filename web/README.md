# Multiband Zolotarev — web front-end

This directory contains:

* `src/` — a C++ port of `../multiband_synthesis.py` (Chebyshev basis,
  sign enumeration, differential-correction LP loop, HiGHS back-end).
* `ui/`  — a static single-page site (`index.html`, `app.js`,
  `style.css`, `presets.js`, `solver_worker.js`) that talks to the
  solver through a Web Worker, plus the paper's two example plots as a
  pre-computed gallery.
* `CMakeLists.txt` that builds either a native smoke-test binary
  (`zolo_test`) or the Emscripten/WebAssembly module (`solver.js` +
  `solver.wasm`) that the UI consumes.

## Prerequisites

| Target                | Tooling                                                                       |
| --------------------- | ----------------------------------------------------------------------------- |
| Native smoke-test     | CMake ≥ 3.20, a C++17 compiler, Git (FetchContent pulls HiGHS v1.7.2)         |
| WebAssembly build     | Everything above **plus** [emsdk](https://emscripten.org/docs/getting_started/downloads.html) activated in the shell (`emcmake`, `em++` on PATH) |
| GitHub Pages hosting  | A GitHub repo with the `ui/` folder tracked                                   |

> **Windows note.** MSYS2/MinGW ucrt64's `ld` can return error 116 while
> linking the large static HiGHS archive.  The Emscripten build uses
> LLD and is unaffected.  On Windows, prefer the WASM build, or do the
> native smoke test on Linux / macOS / WSL.

## Building the WASM solver

```sh
# one-time, after installing emsdk and running  `source ./emsdk_env.sh`
cd web
emcmake cmake -S . -B build-wasm -DCMAKE_BUILD_TYPE=Release
cmake --build build-wasm -j
# produces ui/solver.js and ui/solver.wasm
```

`CMakeLists.txt` drops the outputs directly into `ui/` so you can serve
the directory as-is.

## Running locally

Any static HTTP server will do (the Web Worker and `fetch`
can't load from `file://`):

```sh
cd ui
python -m http.server 8000          # or: npx serve .
open http://localhost:8000
```

Load one of the gallery presets to populate the form, then hit
**Synthesise filter**.  A first solve takes ~0.5 s to instantiate the
WASM module; subsequent solves typically complete in a few seconds.

## Deploying to GitHub Pages

Simplest path — serve the `ui/` directory straight from a repo:

1. Commit the pre-built `ui/solver.js` and `ui/solver.wasm` alongside the
   rest of `ui/` (GitHub Pages is purely static; it won't run your
   Emscripten build).
2. On GitHub, **Settings → Pages** →
   *Source: Deploy from a branch*, *Branch: main*, *Folder: `/web/ui`*.
3. Wait ~1 min for the first deployment; the URL will be
   `https://<user>.github.io/<repo>/`.

If you prefer to keep the built artefacts out of the main branch, add a
small GitHub Actions workflow (`.github/workflows/pages.yml`) that:

```yaml
steps:
  - uses: actions/checkout@v4
  - uses: mymindstorm/setup-emsdk@v14
  - run: emcmake cmake -S web -B build-wasm && cmake --build build-wasm -j
  - uses: actions/upload-pages-artifact@v3
    with: { path: web/ui }
  - uses: actions/deploy-pages@v4
```

This builds the WASM in CI, uploads just `ui/` as the Pages artefact,
and keeps the repo clean.

## Architecture

```
 index.html ─┬─ presets.js      # paper-example form presets
             ├─ app.js          # DOM + form + plotly charts
             └─ solver_worker.js
                    └── solver.js     ← Emscripten glue
                        └── solver.wasm
                             ├── zolo::solve_json(...)
                             │    └── HiGHS LP (linked in)
                             └── Chebyshev basis, sign enumeration,
                                 Remez exchange, differential correction
```

Data flow: the UI builds a JSON spec, posts it to the worker; the
worker calls the WASM's embind-exposed `solve_json`, which returns a
JSON string with the optimum polynomials *and* a pre-evaluated
response on a dense `ω` grid.  The page plots that grid with Plotly.

## Numerical parity with the Python reference

`src/solver.cpp` mirrors `multiband_synthesis.py` step-for-step:

| Python                         | C++                                |
| ------------------------------ | ---------------------------------- |
| `_cheb_basis_row`              | `cheb_basis`                       |
| `_cheb_eval`                   | `cheb_eval`                        |
| `_build_lp`                    | `build_and_solve_lp`               |
| `_probe_lp`                    | `probe_lp`                         |
| `_solve_signed_diffcorr`       | `solve_signed`                     |
| `solve_zolotarev`              | `zolo::solve_zolotarev`            |
| `scipy.optimize.linprog` HiGHS | HiGHS C++ API (exact same solver)  |

Both call the same HiGHS dual-simplex under the hood, so results should
match to LP tolerance (~1e-6 relative on M).  The `zolo_test` native
binary exists precisely to cross-check this on a platform where the
MinGW linker cooperates.
