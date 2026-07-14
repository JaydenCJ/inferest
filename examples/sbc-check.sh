#!/usr/bin/env bash
# SBC sanity check: what can small boards really do? Includes a custom
# device — the whole point is estimating hardware you do not own yet.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${INFEREST_BIN:-$ROOT/inferest}"
if [ ! -x "$BIN" ]; then
  echo "building inferest..."
  (cd "$ROOT" && go build -o "$BIN" ./cmd/inferest)
fi

echo "==> 1B class on a Raspberry Pi 5: the honest numbers"
"$BIN" estimate --device raspberry-pi-5 --model 1b --quant q4 --context 4096

echo
echo "==> Pi 5 vs Jetson Orin Nano for the 3B class"
"$BIN" compare --devices raspberry-pi-5,jetson-orin-nano-8gb --model 3b --quant q4 --context 4096

echo
echo "==> A board that is not in the presets? Describe it with three flags."
echo "    (hypothetical 16 GiB board: 50 GB/s LPDDR5, 2 TFLOPS fp16)"
"$BIN" estimate --model 3b --quant q4 --context 4096 \
  --bandwidth 50 --tflops 2 --memory-gb 16
