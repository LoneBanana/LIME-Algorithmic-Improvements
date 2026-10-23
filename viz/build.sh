#!/usr/bin/env bash
# Build the C++/OpenGL renderer.
#   viz/build.sh test   native unit tests for the geometry (clang++ or g++)
#   viz/build.sh wasm   WebAssembly + WebGL2 build into web/public/viz/ (needs Emscripten: brew install emscripten)
set -euo pipefail
cd "$(dirname "$0")"
mode="${1:-wasm}"
mkdir -p build

if [[ "$mode" == "test" ]]; then
  "${CXX:-clang++}" -std=c++17 -O2 -Wall -Wextra -Wpedantic src/geometry.cpp tests/test_geometry.cpp -o build/test_geometry
  ./build/test_geometry
elif [[ "$mode" == "wasm" ]]; then
  out=../web/public/viz
  mkdir -p "$out"
  em++ -std=c++17 -O3 -Wall -Wextra src/geometry.cpp src/engine.cpp -o "$out/viz.js" \
    -sMODULARIZE=1 -sEXPORT_NAME=createVizModule -sENVIRONMENT=web \
    -sMIN_WEBGL_VERSION=2 -sMAX_WEBGL_VERSION=2 \
    -sALLOW_MEMORY_GROWTH=1 -sFILESYSTEM=0 \
    -sEXPORTED_FUNCTIONS=_malloc,_free \
    -sEXPORTED_RUNTIME_METHODS=cwrap,HEAPF32
  ls -l "$out"
else
  echo "usage: $0 [test|wasm]" >&2
  exit 2
fi
