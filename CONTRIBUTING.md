# Contributing to inferest

Issues, discussions and pull requests are all welcome.

## Getting started

You need Go ≥1.22; nothing else — the module has zero dependencies.

```bash
git clone https://github.com/JaydenCJ/inferest && cd inferest
go build ./...
go test ./...
bash scripts/smoke.sh
```

`scripts/smoke.sh` builds the binary and drives every subcommand against
built-in presets and a custom device, asserting on real output and exit
codes; it must finish by printing `SMOKE OK`.

## Before you open a pull request

1. `gofmt -l .` reports nothing (formatting is enforced).
2. `go vet ./...` passes with no findings.
3. `go test ./...` passes (88 deterministic tests, no network).
4. `bash scripts/smoke.sh` prints `SMOKE OK`.
5. Add tests for behavior changes; keep the math in pure, unit-testable
   modules (`roofline`, `model`, `quant`, `device` never touch I/O — only
   `cli` and `render` do).

## Ground rules

- Keep dependencies at zero; adding one needs strong justification in the PR.
- No network calls, ever — inferest is a calculator. No telemetry.
- Preset data is sourced, not guessed: new device numbers come from public
  spec sheets (dense FP16, not sparse marketing figures), new model
  geometries must pass the derived-parameter cross-check within 2%.
- Changing an efficiency band or the overhead model needs calibration
  evidence in the PR (measured tokens/s vs the estimate, docs/method.md
  updated to match).
- Code comments and doc comments are written in English.
- Determinism first: identical inputs must produce byte-identical reports.

## Reporting bugs

Include the output of `inferest version`, the full command you ran, and —
for "the estimate is off" reports — the measured tokens/s, the runtime and
its settings (batch size, context fill), and the exact hardware, so the gap
can be attributed to an input, a band, or the model itself.

## Security

Please do not open public issues for security problems; use GitHub's
private vulnerability reporting on this repository instead.
