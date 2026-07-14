# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-12

### Added

- Closed-form roofline estimator for single-stream LLM inference: decode
  tokens/s as min(bandwidth bound, compute bound) with the KV cache re-read
  per token, prefill tokens/s and time-to-first-token as min(compute bound,
  amortized-weight bandwidth bound), all pure functions of the inputs.
- Conservative / expected / optimistic efficiency bands (55–85% of
  spec-sheet bandwidth, 25–55% model-FLOPs utilization), collapsible to a
  single user value with `--bw-eff` / `--mfu`.
- Memory-fit model: weights (effective bits per weight, block overhead
  included), KV cache per context token, fixed + weight-fraction +
  compute-buffer overhead; solves the largest context that fits.
- `estimate` subcommand with text, stable JSON (`schema_version: 1`) and
  PR-pasteable Markdown output.
- `compare` subcommand: the same estimate across several devices in caller
  order, with non-fitting rows dashed out instead of showing fantasy numbers.
- `fit` subcommand: memory verdict with exit code 1 on "does not fit" (a
  shell-gate contract) and the widest quantization that would fit.
- 19 device presets from public spec sheets (data-center and consumer GPUs,
  unified-memory SoCs, SBCs, DDR4/DDR5 desktops), every number overridable
  per run; fully custom devices via `--bandwidth/--tflops/--memory-gb`.
- 11 model geometry presets (1B–70B dense classes incl. MHA vs GQA
  generations, plus two MoE classes with total/active split); dense presets
  are cross-checked against parameter counts derived from their own
  geometry; fully custom geometry via flags.
- Quantization tables with *effective* bits per weight (q2–q8, f16/bf16,
  f32) and KV-cache precisions (f32/f16/q8/q4).
- `devices`, `models`, `quants` listings in text and JSON.
- Runnable examples (`examples/gpu-shortlist.sh`, `examples/sbc-check.sh`)
  and a full derivation/calibration write-up (`docs/method.md`).
- 88 deterministic offline tests (exact closed-form checks on a toy
  geometry, physics checks on realistic presets, in-process CLI
  integration) and `scripts/smoke.sh`.

[0.1.0]: https://github.com/JaydenCJ/inferest/releases/tag/v0.1.0
