#!/usr/bin/env bash
# Builds web/site/smartdnn.js and smartdnn.wasm with Emscripten (em++ on PATH), embedding web/model.bin.
set -euo pipefail
cd "$(dirname "$0")/.."

em++ -std=c++17 -O3 -I . web/mnist_wasm.cpp -o web/site/smartdnn.js \
    -sMODULARIZE -sEXPORT_ES6 -sEXPORT_NAME=createSmartDNN -sENVIRONMENT=web,node \
    -sALLOW_MEMORY_GROWTH -sEXPORTED_FUNCTIONS=_load_model,_predict,_malloc,_free \
    -sEXPORTED_RUNTIME_METHODS=HEAPF32 --embed-file web/model.bin@/model.bin
