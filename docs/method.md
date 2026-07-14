# The math behind inferest

Every number inferest prints is derived from three device figures, a model
geometry, and a quantization choice. This document is the full derivation,
the unit conventions, the calibration evidence for the efficiency bands, and
an honest list of what the model ignores.

## 1. Unit conventions

| Quantity | Unit | Why |
|---|---|---|
| Memory capacity | GiB (binary, 2³⁰) | VRAM and unified memory are marketed in binary units |
| Memory bandwidth | GB/s (decimal, 10⁹) | spec sheets quote decimal GB/s |
| Compute | TFLOPS (decimal, 10¹²), dense FP16 | inference kernels run half precision; sparse marketing numbers are ~2× dense |

Mixing binary and decimal shifts every estimate by ~7%, so the conversion
functions live in one place (`internal/device`) and are pinned by tests.

## 2. Model geometry

A decoder-only transformer is fully described (for our purposes) by: layers
`L`, hidden width `d`, query heads `H`, KV heads `H_kv` (GQA), per-head
dimension `h` (usually `d/H`, sometimes explicit), FFN width `f`, vocabulary
`V`, and — for MoE — expert count `E` with `A` routed per token.

From geometry alone, the parameter count is derived as:

```
attention/layer = d·(H·h)      (Q)  +  (H·h)·d    (O)
                + 2·d·(H_kv·h) (K and V)
ffn/layer       = 3·d·f·(E or 1)          gated FFN: up, gate, down
router/layer    = d·E                     MoE only
embeddings      = V·d  (×2 unless tied to the output head)
```

Every dense preset's declared parameter count must match this derivation
within 2% (enforced by tests). This keeps the preset table honest and
catches typos the moment they are introduced.

The KV cache costs, per context token:

```
kv_bytes/token = 2 · L · H_kv · h · bytes_per_element
```

For the 8B GQA class at f16 that is 2·32·8·128·2 = 131,072 B = 128 KiB per
token — the number that decides long-context feasibility.

## 3. Decode: why bandwidth wins

Generating one token with batch size 1 must stream **every active weight
byte** and **the entire KV cache** past the compute units:

```
bytes/token  = active_params · bits_per_weight / 8  +  kv_bytes/token · T
flops/token  = 2 · active_params  +  4 · L · (H·h) · T
decode t/s   = min( eff_bw · BW / bytes_per_token ,
                    mfu · FLOPS / flops_per_token )
```

where `T` is the current context fill. The arithmetic intensity of decode is
~2 FLOPs per byte at 16-bit, ~4 at 4-bit — while modern GPUs offer 100–500
FLOPs per byte of bandwidth. That two-orders-of-magnitude gap is why decode
is bandwidth-bound on every realistic device, and why inferest reports the
compute *headroom* rather than pretending the bounds compete.

MoE models split the two roles: **footprint** charges all `E` experts,
**traffic and FLOPs** charge only the `A` routed ones — which is exactly why
a 47B-total/13B-active model decodes like a 13B but must fit like a 47B.

## 4. Prefill: why compute wins

The prompt is processed as one batch, so weight traffic amortizes across all
`P` positions while FLOPs do not:

```
flops/token  = 2 · active_params  +  2 · L · (H·h) · P     (causal ½ · 4LwP)
bytes/token  = weight_bytes / P  +  kv_bytes/token          (weights amortized)
prefill t/s  = min( mfu · FLOPS / flops_per_token ,
                    eff_bw · BW / bytes_per_token )
TTFT         = P / prefill_t/s
```

At `P = 1024` the weight traffic per token is ~1000× smaller than in decode,
so the compute bound takes over. This is why a Raspberry Pi that decodes at
2 t/s still needs minutes of prefill for a long prompt — and why inferest
reports both numbers separately instead of a single misleading "tokens/s".

## 5. Memory-fit model

```
total = weights · (1 + 2%)              fragmentation + dequant scratch
      + kv_bytes/token · context
      + 8 B · d · context               activation/compute buffers
      + 256 MiB                         runtime, logits, tokenizer tables
```

The same expression is solved for the largest integer `context` that fits,
so the verdict and the "max context" figure can never disagree (pinned by a
self-consistency test). The overhead terms are deliberately simple and
documented here rather than hidden; they are constants in
`internal/roofline/roofline.go`.

## 6. Efficiency bands and calibration

Theoretical peaks are never reached. inferest ships two bands:

| Band | Conservative | Expected | Optimistic | Applies to |
|---|---|---|---|---|
| Bandwidth efficiency | 0.55 | 0.70 | 0.85 | decode, KV traffic |
| Model FLOPs utilization | 0.25 | 0.40 | 0.55 | prefill, compute bound |

Sanity anchors against publicly reported single-stream numbers (8B-class
dense model at 4-bit, f16 cache, short context):

| Device | inferest expected | commonly reported |
|---|---|---|
| 24 GiB / 1008 GB/s GPU | ≈156 t/s | 120–160 t/s |
| 546 GB/s unified SoC | ≈85 t/s | 70–90 t/s |
| 273 GB/s unified SoC | ≈42 t/s | 35–50 t/s |
| 17.1 GB/s SBC | ≈2.7 t/s | 2–3 t/s |

`--bw-eff` and `--mfu` collapse a band to a single value when you have
measured your own stack's efficiency and want the model to use it.

## 7. What the model ignores (on purpose)

- **Batching beyond 1.** Single-stream only; throughput-oriented serving
  with continuous batching moves decode toward the compute roof.
- **Speculative decoding**, draft models, and self-speculation — these
  multiply effective t/s by an acceptance-rate factor inferest cannot know.
- **Tensor/pipeline parallelism.** One device at a time; interconnects are
  their own roofline.
- **Attention-kernel subtleties** (flash attention IO-awareness, paged KV):
  bounded by the same weight+cache traffic; the bands absorb the difference.
- **Quality.** inferest tells you a 2-bit 70B *fits and how fast it runs*,
  not whether you should want one.

An estimate is a hypothesis, not a benchmark. When a measured number lands
outside the conservative–optimistic range, one of the *inputs* is wrong —
which is precisely the argument inferest is designed to make checkable.
