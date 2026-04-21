# Build the multiband Zolotarev solver, native or Emscripten.
#
# Native smoke test:
#   cmake -S . -B build-native -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-native -j
#   ./build-native/zolo_test
#
# WebAssembly (requires emsdk activated):
#   emcmake cmake -S . -B build-wasm -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-wasm -j
#   # produces ui/solver.js + ui/solver.wasm
cmake_minimum_required(VERSION 3.20)
project(zolotarev_web CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)

# ----- HiGHS ---------------------------------------------------------------
# HiGHS is the LP backend.  We pull a specific stable tag so the build is
# reproducible.  Tests off, shared libs off for a clean static link.
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(FAST_BUILD ON CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(PYTHON OFF CACHE BOOL "" FORCE)
set(CSHARP OFF CACHE BOOL "" FORCE)
set(FORTRAN OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  highs
  GIT_REPOSITORY https://github.com/ERGO-Code/HiGHS.git
  GIT_TAG v1.7.2
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(highs)

# ----- nlohmann/json (header-only, vendored) -------------------------------
# We include a single-header copy under third_party/ to avoid pulling another
# dependency.  See third_party/nlohmann/json.hpp.

# ----- Solver library ------------------------------------------------------
add_library(zolo STATIC
  src/solver.cpp
  src/solver_json.cpp
)
target_include_directories(zolo
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)
target_link_libraries(zolo PUBLIC highs)
target_compile_options(zolo PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wno-unused-parameter>
)

# ----- Native smoke test ---------------------------------------------------
if (NOT EMSCRIPTEN)
  add_executable(zolo_test src/zolo_test.cpp)
  target_link_libraries(zolo_test PRIVATE zolo)
endif()

# ----- Emscripten build ----------------------------------------------------
if (EMSCRIPTEN)
  # Compile solver sources DIRECTLY into the executable rather than going
  # through the static library `zolo`.  The EMSCRIPTEN_BINDINGS macro in
  # solver_json.cpp registers the JS-visible `solve_json` via a C++ static
  # constructor; if that object file lives inside a static archive, LTO
  # drops the registration because nothing references it.  Putting the
  # sources on the executable keeps the constructor in the final link.
  add_executable(solver
    src/solver_wasm_main.cpp
    src/solver.cpp
    src/solver_json.cpp
  )
  target_include_directories(solver PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  target_link_libraries(solver PRIVATE highs)
  set_target_properties(solver PROPERTIES OUTPUT_NAME "solver")
  target_compile_options(solver PRIVATE --bind)
  target_link_options(solver PRIVATE
    "-sMODULARIZE=1"
    "-sEXPORT_NAME=createSolverModule"
    "-sENVIRONMENT=web,worker"
    "-sALLOW_MEMORY_GROWTH=1"
    "-sINITIAL_MEMORY=67108864"   # 64 MB initial heap
    "-sEXPORTED_RUNTIME_METHODS=['ccall','cwrap']"
    "-sDISABLE_EXCEPTION_CATCHING=0"  # embind throws on JSON errors
    "--bind"
    "-O3"
  )
  # Drop the final solver.js + solver.wasm next to the HTML UI.
  set_target_properties(solver PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ui
  )
endif()
