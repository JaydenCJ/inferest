#!/usr/bin/env bash
# GPU buying shortlist: compare candidate cards for the model you actually
# plan to run, then gate the decision on the memory verdict — before
# spending a cent or downloading a gigabyte.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${INFEREST_BIN:-$ROOT/inferest}"
if [ ! -x "$BIN" ]; then
  echo "building inferest..."
  (cd "$ROOT" && go build -o "$BIN" ./cmd/inferest)
fi

echo "==> Daily driver: 8B class at 4-bit, 8k context"
"$BIN" compare --devices rtx-4090,rtx-4080,rtx-3090,rx-7900-xtx --model 8b --quant q4

echo
echo "==> Stretch goal: 70B class at 4-bit — who can even hold it?"
"$BIN" compare --devices rtx-4090,rtx-3090,apple-m4-max,a100-80gb --model 70b --quant q4

echo
echo "==> Purchase gate: does a used 3090 carry the 24B class at 16k context?"
if "$BIN" fit --device rtx-3090 --model 24b --quant q4 --context 16384; then
  echo "gate: PASS — shortlist it"
else
  echo "gate: FAIL — drop it from the shortlist"
fi
