#!/usr/bin/env bash
# Builds web/gpt/site/chargpt.js and chargpt.wasm with Emscripten (em++ on PATH), embedding the trained model and vocabulary.
set -euo pipefail
cd "$(dirname "$0")/../.."

# model.bin is the examples/CharGPT.cpp Tiny Shakespeare model saved with WeightFormat::Int8; vocab.txt is that text's distinct characters in byte order.
em++ -std=c++17 -O3 -msimd128 -I . web/gpt/gpt_wasm.cpp -o web/gpt/site/chargpt.js \
    -sMODULARIZE -sEXPORT_ES6 -sEXPORT_NAME=createCharGpt -sENVIRONMENT=web,node -sALLOW_MEMORY_GROWTH \
    -sEXPORTED_FUNCTIONS=_gpt_load,_gpt_reset,_gpt_next,_gpt_score -sEXPORTED_RUNTIME_METHODS=ccall \
    --embed-file web/gpt/model.bin@/model.bin --embed-file web/gpt/vocab.txt@/vocab.txt
