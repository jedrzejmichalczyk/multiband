// Web worker wrapping the Emscripten-generated solver module.
// solver.js is produced by the CMake/Emscripten build; it exposes a
// Module.solve_json(spec_json) -> result_json via embind.
//
// Messaging:
//   main ->  postMessage(spec)            (spec = plain JS object)
//   wrk  ->  postMessage({ok, result?, error?})
importScripts("solver.js");

let readyPromise = null;

function ensureReady() {
  if (!readyPromise) {
    // createSolverModule is provided by the -sEXPORT_NAME= flag in CMakeLists.
    readyPromise = createSolverModule().then((Module) => ({ Module }));
  }
  return readyPromise;
}

self.onmessage = async (ev) => {
  try {
    const { Module } = await ensureReady();
    const json = Module.solve_json(JSON.stringify(ev.data));
    const result = JSON.parse(json);
    self.postMessage({ ok: true, result });
  } catch (e) {
    self.postMessage({ ok: false, error: String(e.message ?? e) });
  }
};
