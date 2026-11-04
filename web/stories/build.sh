#!/usr/bin/env bash
# Builds web/stories/site/storygpt.js and storygpt.wasm with Emscripten (em++ on PATH), embedding the trained model and tokenizer.
set -euo pipefail
cd "$(dirname "$0")/../.."

# model.bin is the best web/stories/train.cpp checkpoint saved with WeightFormat::Int8; tokenizer.txt holds its BPE merges.
em++ -std=c++17 -O3 -msimd128 -I . web/stories/story_wasm.cpp -o web/stories/site/storygpt.js \
    -sMODULARIZE -sEXPORT_ES6 -sEXPORT_NAME=createStoryGpt -sENVIRONMENT=web,node -sALLOW_MEMORY_GROWTH \
    -sEXPORTED_FUNCTIONS=_story_load,_story_begin,_story_next,_story_piece,_story_score -sEXPORTED_RUNTIME_METHODS=ccall,HEAPU8 \
    --embed-file web/stories/model.bin@/model.bin --embed-file web/stories/tokenizer.txt@/tokenizer.txt
