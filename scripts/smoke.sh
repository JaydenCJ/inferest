#!/usr/bin/env bash
# End-to-end smoke test for inferest: builds the binary, runs every
# subcommand against built-in presets and a custom device, and asserts on
# real CLI output and exit codes. No network, idempotent, finishes in
# seconds.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

fail() {
  echo "SMOKE FAIL: $*" >&2
  exit 1
}

BIN="$WORKDIR/inferest"

echo "1. build"
(cd "$ROOT" && go build -o "$BIN" ./cmd/inferest) || fail "go build failed"

echo "2. version matches the manifest"
"$BIN" version | grep -qx "inferest 0.1.0" || fail "version mismatch"

echo "3. estimate: text report for a preset pair"
OUT="$("$BIN" estimate --device rtx-4090 --model 8b --quant q4)"
echo "$OUT" | grep -q "inferest estimate — 8b @ q4 on rtx-4090" || fail "estimate header missing"
echo "$OUT" | grep -q "FITS" || fail "8b @ q4 must fit a 24 GiB card"
echo "$OUT" | grep -q "bound: memory bandwidth" || fail "decode must be bandwidth-bound"
echo "$OUT" | grep -q "prefill (1,024-token prompt)" || fail "prefill section missing"

echo "4. estimate: JSON is machine-readable and versioned"
JSON="$("$BIN" estimate --device rtx-4090 --model 8b --format json)"
echo "$JSON" | grep -q '"tool": "inferest"' || fail "json envelope missing"
echo "$JSON" | grep -q '"schema_version": 1' || fail "json schema version missing"
echo "$JSON" | grep -q '"kv_bytes_per_token": 131072' || fail "8b KV/token should be 131072 B"

echo "5. estimate: markdown emits tables"
"$BIN" estimate --device rtx-4090 --model 8b --format markdown \
  | grep -q '| Decode t/s | Conservative | Expected | Optimistic |' \
  || fail "markdown decode table missing"

echo "6. compare: one row per device, caller order"
CMP="$("$BIN" compare --devices rtx-4090,apple-m4-pro,raspberry-pi-5 --model 8b --quant q4)"
echo "$CMP" | grep -q "raspberry-pi-5" || fail "compare row missing"
[ "$(echo "$CMP" | grep -c 'rtx-4090\|apple-m4-pro\|raspberry-pi-5')" -eq 3 ] \
  || fail "compare should have exactly three device rows"

echo "7. custom device: estimate hardware nobody owns yet"
"$BIN" estimate --model 8b --bandwidth 200 --tflops 20 --memory-gb 16 \
  | grep -q "on custom" || fail "custom device estimate failed"

echo "8. fit: exit 0 when it fits, 1 when it does not (shell-gate contract)"
"$BIN" fit --device rtx-4090 --model 8b --quant q4 >/dev/null \
  || fail "8b @ q4 should fit and exit 0"
set +e
"$BIN" fit --device rtx-4090 --model 70b --quant q4 >"$WORKDIR/fit.txt"
CODE=$?
set -e
[ "$CODE" -eq 1 ] || fail "70b @ q4 on 24 GiB should exit 1, got $CODE"
grep -q "DOES NOT FIT" "$WORKDIR/fit.txt" || fail "no-fit verdict missing"

echo "9. fit: suggests the widest quantization that would fit"
set +e
SUGGEST="$("$BIN" fit --device a100-80gb --model 70b --quant f16)" # exits 1 by design
set -e
echo "$SUGGEST" | grep -q "widest quantization that fits at this context: q8" \
  || fail "fit should suggest q8 for 70b f16 on 80 GiB"

echo "10. listings: devices, models, quants"
"$BIN" devices | grep -q "raspberry-pi-5" || fail "devices listing incomplete"
"$BIN" models | grep -q "moe-8x7b" || fail "models listing incomplete"
"$BIN" quants | grep -q "4.50" || fail "quants listing should show effective bits"

echo "11. usage errors exit 2"
set +e
"$BIN" estimate --device gtx-9090 --model 8b >/dev/null 2>&1
[ $? -eq 2 ] || fail "unknown device should exit 2"
"$BIN" estimate --device rtx-4090 --model 8b --format yaml >/dev/null 2>&1
[ $? -eq 2 ] || fail "bad --format should exit 2"
set -e

echo "SMOKE OK"
