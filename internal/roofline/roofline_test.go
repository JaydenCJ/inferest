// Tests for the roofline math. Strategy: a hand-checkable toy geometry with
// efficiency overrides pinned to 1.0 verifies the closed forms exactly;
// realistic presets then verify the qualitative physics (bandwidth binds
// decode, compute binds prefill, MoE splits footprint from traffic).
package roofline

import (
	"math"
	"reflect"
	"testing"

	"github.com/JaydenCJ/inferest/internal/device"
	"github.com/JaydenCJ/inferest/internal/model"
	"github.com/JaydenCJ/inferest/internal/quant"
)

// toyInputs returns a fully-specified request whose every derived number
// fits in a pocket calculator: 1B params, q8 (8.5 bpw → 1.0625 GB of
// weights), 20 kB of KV per token, 1000 GB/s, 100 TFLOPS.
func toyInputs(t *testing.T) Inputs {
	t.Helper()
	q8, err := quant.Lookup("q8")
	if err != nil {
		t.Fatalf("quant.Lookup: %v", err)
	}
	f16, err := quant.LookupKV("f16")
	if err != nil {
		t.Fatalf("quant.LookupKV: %v", err)
	}
	return Inputs{
		Device: device.Device{Name: "toy-dev", Kind: "gpu", MemoryGiB: 24, BandwidthGBs: 1000, TFLOPSFP16: 100},
		Model: model.Geometry{
			Name: "toy", ParamsB: 1.0, Layers: 10, DModel: 1000, Heads: 10, KVHeads: 5,
			HeadDim: 100, FFN: 4000, Vocab: 50000,
		},
		Weights: q8, KVCache: f16,
		Context: 4096, Prompt: 1024,
		BandwidthEff: 1.0, MFU: 1.0, // exact math: no efficiency scaling
	}
}

// presetInputs returns a realistic request built from the shipped presets.
func presetInputs(t *testing.T, dev, mdl, q string) Inputs {
	t.Helper()
	d, err := device.Lookup(dev)
	if err != nil {
		t.Fatalf("device.Lookup(%s): %v", dev, err)
	}
	g, err := model.Lookup(mdl)
	if err != nil {
		t.Fatalf("model.Lookup(%s): %v", mdl, err)
	}
	w, err := quant.Lookup(q)
	if err != nil {
		t.Fatalf("quant.Lookup(%s): %v", q, err)
	}
	kv, err := quant.LookupKV("f16")
	if err != nil {
		t.Fatalf("quant.LookupKV: %v", err)
	}
	return Inputs{Device: d, Model: g, Weights: w, KVCache: kv, Context: 8192, Prompt: 1024}
}

func almostEqual(got, want, relTol float64) bool {
	return math.Abs(got-want) <= relTol*math.Abs(want)
}

func TestFootprintComponentsAreExact(t *testing.T) {
	est, err := New(toyInputs(t))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if want := 1e9 * 8.5 / 8; est.Memory.WeightBytes != want {
		t.Fatalf("WeightBytes = %g, want %g", est.Memory.WeightBytes, want)
	}
	// 2 × 10 layers × 5 kv-heads × 100 head-dim × 2 B = 20000 B/token.
	if est.Memory.KVBytesPerToken != 20000 {
		t.Fatalf("KVBytesPerToken = %g, want 20000", est.Memory.KVBytesPerToken)
	}
	if want := 20000.0 * 4096; est.Memory.KVBytes != want {
		t.Fatalf("KVBytes = %g, want %g", est.Memory.KVBytes, want)
	}
}

func TestDecodeBandwidthBoundExactValue(t *testing.T) {
	est, err := New(toyInputs(t))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	// At empty context with eff=1.0: 1e12 B/s ÷ 1.0625e9 B/token.
	first := est.Decode.Points[0]
	if first.ContextTokens != 0 {
		t.Fatalf("first decode point should be empty context, got %d", first.ContextTokens)
	}
	want := 1e12 / 1.0625e9
	if !almostEqual(first.TPS.Expected, want, 1e-12) {
		t.Fatalf("decode TPS at empty context = %g, want %g", first.TPS.Expected, want)
	}
}

func TestDecodeAccountsForKVCacheReads(t *testing.T) {
	est, err := New(toyInputs(t))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	// At 4096 tokens of history the whole cache is re-read per token:
	// bytes/token = 1.0625e9 + 4096 × 20000.
	last := est.Decode.Points[len(est.Decode.Points)-1]
	if want := 1.0625e9 + 4096*20000; last.BytesPerToken != want {
		t.Fatalf("BytesPerToken at full context = %g, want %g", last.BytesPerToken, want)
	}
	if !almostEqual(last.TPS.Expected, 1e12/(1.0625e9+4096*20000), 1e-12) {
		t.Fatalf("decode TPS at full context wrong: %g", last.TPS.Expected)
	}
}

func TestDecodeComputeBoundBindsWhenComputeIsTiny(t *testing.T) {
	in := toyInputs(t)
	in.Device.TFLOPSFP16 = 0.001 // 1 GFLOP/s: compute becomes the wall
	est, err := New(in)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	first := est.Decode.Points[0]
	// flops/token at empty context = 2 × 1e9; bound = 1e9 / 2e9 = 0.5 t/s.
	if !almostEqual(first.TPS.Expected, 0.5, 1e-12) {
		t.Fatalf("compute-bound decode TPS = %g, want 0.5", first.TPS.Expected)
	}
	if est.Decode.BandwidthLimited {
		t.Fatal("decode should be compute-limited on a 1 GFLOP/s device")
	}
}

func TestDecodeTPSMonotonicallyDecreasesWithContextFill(t *testing.T) {
	est, err := New(presetInputs(t, "rtx-4090", "8b", "q4"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	pts := est.Decode.Points
	if len(pts) != 3 {
		t.Fatalf("expected 3 decode points (0, ctx/2, ctx), got %d", len(pts))
	}
	for i := 1; i < len(pts); i++ {
		if pts[i].TPS.Expected >= pts[i-1].TPS.Expected {
			t.Fatalf("decode TPS should fall as context fills: %g then %g",
				pts[i-1].TPS.Expected, pts[i].TPS.Expected)
		}
	}
}

func TestDecodePointsDeduplicateAtTinyContext(t *testing.T) {
	in := toyInputs(t)
	in.Context, in.Prompt = 1, 1 // positions 0, 0, 1 → two unique points
	est, err := New(in)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if len(est.Decode.Points) != 2 {
		t.Fatalf("expected 2 deduplicated points, got %d", len(est.Decode.Points))
	}
}

func TestBandsOrderedConservativeToOptimistic(t *testing.T) {
	est, err := New(presetInputs(t, "rtx-4090", "8b", "q4"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	for _, p := range est.Decode.Points {
		if !(p.TPS.Conservative < p.TPS.Expected && p.TPS.Expected < p.TPS.Optimistic) {
			t.Fatalf("band out of order at context %d: %+v", p.ContextTokens, p.TPS)
		}
	}
	pf := est.Prefill
	if !(pf.TTFTSeconds.Conservative > pf.TTFTSeconds.Expected && pf.TTFTSeconds.Expected > pf.TTFTSeconds.Optimistic) {
		t.Fatalf("TTFT band should invert (slower TPS → longer wait): %+v", pf.TTFTSeconds)
	}
}

func TestEfficiencyOverridesCollapseTheirBands(t *testing.T) {
	in := presetInputs(t, "rtx-4090", "8b", "q4")
	in.BandwidthEff = 0.8
	est, err := New(in)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	p := est.Decode.Points[0]
	if p.TPS.Conservative != p.TPS.Expected || p.TPS.Expected != p.TPS.Optimistic {
		t.Fatalf("override should collapse the band, got %+v", p.TPS)
	}
	want := 0.8 * in.Device.BandwidthBytesPerSec() / p.BytesPerToken
	if !almostEqual(p.TPS.Expected, want, 1e-12) {
		t.Fatalf("overridden decode TPS = %g, want %g", p.TPS.Expected, want)
	}

	in2 := presetInputs(t, "rtx-4090", "8b", "q4")
	in2.MFU = 0.5
	est2, err := New(in2)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if pf := est2.Prefill; pf.TPS.Conservative != pf.TPS.Optimistic {
		t.Fatalf("MFU override should collapse the prefill band, got %+v", pf.TPS)
	}
}

func TestRealisticGPUDecodeIsBandwidthLimited(t *testing.T) {
	// The core claim of the tool: on real hardware, decode is a memory
	// problem, not a compute problem.
	est, err := New(presetInputs(t, "rtx-4090", "8b", "q4"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if !est.Decode.BandwidthLimited {
		t.Fatal("8b @ q4 on a 4090 must be bandwidth-limited")
	}
	if est.Decode.ComputeHeadroom < 5 {
		t.Fatalf("compute headroom should be large, got %.1fx", est.Decode.ComputeHeadroom)
	}
	last := est.Decode.Points[len(est.Decode.Points)-1]
	want := last.ComputeBound.Expected / last.BandwidthBound.Expected
	if !almostEqual(est.Decode.ComputeHeadroom, want, 1e-12) {
		t.Fatalf("ComputeHeadroom = %g, want ratio of bounds %g", est.Decode.ComputeHeadroom, want)
	}
}

func TestRealisticGPUPrefillIsComputeLimited(t *testing.T) {
	est, err := New(presetInputs(t, "rtx-4090", "8b", "q4"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if !est.Prefill.ComputeLimited {
		t.Fatal("batched prefill on a 4090 must be compute-limited")
	}
	last := est.Decode.Points[len(est.Decode.Points)-1]
	if est.Prefill.TPS.Expected <= last.TPS.Expected {
		t.Fatal("prefill throughput must exceed decode throughput")
	}
}

func TestPrefillTTFTEqualsPromptOverTPS(t *testing.T) {
	est, err := New(presetInputs(t, "rtx-4090", "8b", "q4"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	pf := est.Prefill
	want := float64(pf.PromptTokens) / pf.TPS.Expected
	if !almostEqual(pf.TTFTSeconds.Expected, want, 1e-12) {
		t.Fatalf("TTFT = %g, want prompt/TPS = %g", pf.TTFTSeconds.Expected, want)
	}
}

func TestPrefillExactComputeBoundOnToy(t *testing.T) {
	est, err := New(toyInputs(t))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	// flops/token = 2×1e9 + 2×10×(10×100)×1024 = 2e9 + 2.048e7.
	wantFlops := 2e9 + 2*10*1000*1024.0
	if est.Prefill.FlopsPerToken != wantFlops {
		t.Fatalf("prefill FlopsPerToken = %g, want %g", est.Prefill.FlopsPerToken, wantFlops)
	}
	wantTPS := 100e12 / wantFlops // MFU pinned to 1.0
	if !almostEqual(est.Prefill.TPS.Expected, wantTPS, 1e-12) {
		t.Fatalf("prefill TPS = %g, want %g", est.Prefill.TPS.Expected, wantTPS)
	}
}

func TestMoEDecodeTrafficUsesActiveParamsOnly(t *testing.T) {
	moe := presetInputs(t, "rtx-4090", "moe-8x7b", "q4")
	est, err := New(moe)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	first := est.Decode.Points[0]
	wantActive := moe.Weights.BytesForParams(moe.Model.ActiveParams())
	if first.BytesPerToken != wantActive {
		t.Fatalf("MoE decode bytes/token = %g, want active-only %g", first.BytesPerToken, wantActive)
	}
	// ...while the footprint charges for every expert.
	wantTotal := moe.Weights.BytesForParams(moe.Model.TotalParams())
	if est.Memory.WeightBytes != wantTotal {
		t.Fatalf("MoE weight footprint = %g, want total %g", est.Memory.WeightBytes, wantTotal)
	}
}

func TestFitsVerdictSmallModelOnBigGPU(t *testing.T) {
	est, err := New(presetInputs(t, "rtx-4090", "8b", "q4"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	m := est.Memory
	if !m.Known || !m.Fits {
		t.Fatalf("8b @ q4 must fit in 24 GiB, got %+v", m)
	}
	if m.UsedFraction <= 0 || m.UsedFraction >= 0.5 {
		t.Fatalf("used fraction should be modest, got %g", m.UsedFraction)
	}
}

func TestDoesNotFitVerdict70BOn24GB(t *testing.T) {
	est, err := New(presetInputs(t, "rtx-4090", "70b", "q4"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if est.Memory.Fits {
		t.Fatal("70b @ q4 (≈37 GiB of weights) must not fit in 24 GiB")
	}
	if est.Memory.MaxContext != 0 {
		t.Fatalf("weights alone exceed capacity, MaxContext should be 0, got %d", est.Memory.MaxContext)
	}
}

func TestMaxContextIsSelfConsistent(t *testing.T) {
	// The solved MaxContext must itself fit, and a healthy margin above it
	// must not — the solver and the verdict share one overhead model.
	base := presetInputs(t, "rtx-4090", "8b", "q4")
	est, err := New(base)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	maxCtx := est.Memory.MaxContext
	if maxCtx <= base.Context {
		t.Fatalf("8b @ q4 on 24 GiB should allow more than %d tokens, got %d", base.Context, maxCtx)
	}

	at := base
	at.Context = maxCtx
	atEst, err := New(at)
	if err != nil {
		t.Fatalf("New at max context: %v", err)
	}
	if !atEst.Memory.Fits {
		t.Fatalf("context %d (the solved max) should fit", maxCtx)
	}

	over := base
	over.Context = maxCtx + 1024
	overEst, err := New(over)
	if err != nil {
		t.Fatalf("New over max context: %v", err)
	}
	if overEst.Memory.Fits {
		t.Fatalf("context %d (beyond the solved max) should not fit", over.Context)
	}
}

func TestUnknownCapacityDisablesFitVerdict(t *testing.T) {
	in := toyInputs(t)
	in.Device.MemoryGiB = 0
	est, err := New(in)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if est.Memory.Known {
		t.Fatal("capacity 0 must mean unknown")
	}
	if est.Memory.MaxContext != -1 {
		t.Fatalf("unknown capacity should report MaxContext -1, got %d", est.Memory.MaxContext)
	}
}

func TestKVQuantHalvesCacheFootprint(t *testing.T) {
	f16 := presetInputs(t, "rtx-4090", "8b", "q4")
	q8kv, err := quant.LookupKV("q8")
	if err != nil {
		t.Fatalf("LookupKV: %v", err)
	}
	half := f16
	half.KVCache = q8kv
	a, err := New(f16)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	b, err := New(half)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if b.Memory.KVBytes*2 != a.Memory.KVBytes {
		t.Fatalf("q8 cache should halve KV bytes: %g vs %g", b.Memory.KVBytes, a.Memory.KVBytes)
	}
}

func TestValidateRejectsBadRequests(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*Inputs)
	}{
		{"negative context", func(in *Inputs) { in.Context = -1 }},
		{"zero prompt", func(in *Inputs) { in.Prompt = 0 }},
		{"prompt beyond context", func(in *Inputs) { in.Prompt = in.Context + 1 }},
		{"bandwidth efficiency above 1", func(in *Inputs) { in.BandwidthEff = 1.5 }},
		{"mfu above 1", func(in *Inputs) { in.MFU = 2 }},
		{"broken device", func(in *Inputs) { in.Device.BandwidthGBs = 0 }},
		{"broken model", func(in *Inputs) { in.Model.Layers = 0 }},
	}
	for _, tc := range cases {
		in := toyInputs(t)
		tc.mutate(&in)
		if _, err := New(in); err == nil {
			t.Errorf("%s: New() should fail", tc.name)
		}
	}
}

func TestEstimateIsDeterministic(t *testing.T) {
	// Pure closed-form math: two runs over the same inputs must produce a
	// deeply equal struct — no maps, clocks or randomness anywhere.
	in := presetInputs(t, "apple-m4-max", "moe-30b-a3b", "q5")
	a, err := New(in)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	b, err := New(in)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if !reflect.DeepEqual(a, b) {
		t.Fatal("identical inputs produced different estimates")
	}
}
