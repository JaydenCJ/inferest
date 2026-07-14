// Tests for transformer geometry. The centerpiece is the derived-parameter
// cross-check: every preset's declared size must be reproducible from its
// own layer/width/head numbers, which catches both typos in the table and
// regressions in the derivation formula.
package model

import (
	"math"
	"strings"
	"testing"
)

// relErr returns |got-want|/want.
func relErr(got, want float64) float64 { return math.Abs(got-want) / want }

func TestHeadDimResolution(t *testing.T) {
	// Default: d_model/heads. Real models (fine-grained MoE, some mid-size
	// dense) decouple head_dim from that ratio; the explicit field wins.
	implicit := Geometry{DModel: 4096, Heads: 32}
	if hd := implicit.HeadDimResolved(); hd != 128 {
		t.Fatalf("HeadDimResolved() = %d, want 128 (d_model/heads)", hd)
	}
	explicit := Geometry{DModel: 2048, Heads: 32, HeadDim: 128}
	if hd := explicit.HeadDimResolved(); hd != 128 {
		t.Fatalf("HeadDimResolved() = %d, want explicit 128", hd)
	}
}

func TestDerivedParamsMatchDeclaredForEveryPreset(t *testing.T) {
	// 2% tolerance: presets round published counts to two decimals and the
	// derivation ignores sub-permille terms (biases, routers' norms).
	for _, g := range All() {
		derived := g.DerivedParams() / 1e9
		if e := relErr(derived, g.ParamsB); e > 0.02 {
			t.Errorf("%s: derived %.3fB vs declared %.2fB (rel err %.1f%%)", g.Name, derived, g.ParamsB, e*100)
		}
	}
}

func TestDerivedActiveParamsMatchDeclaredForMoEPresets(t *testing.T) {
	for _, g := range All() {
		if !g.MoE() {
			continue
		}
		derived := g.DerivedActiveParams() / 1e9
		if e := relErr(derived, g.ActiveParamsB); e > 0.02 {
			t.Errorf("%s: derived active %.3fB vs declared %.2fB (rel err %.1f%%)", g.Name, derived, g.ActiveParamsB, e*100)
		}
	}
}

func TestActiveParamsDenseVersusMoE(t *testing.T) {
	dense, _ := Lookup("8b")
	if dense.ActiveParams() != dense.TotalParams() {
		t.Fatalf("dense model: active %g != total %g", dense.ActiveParams(), dense.TotalParams())
	}
	g, _ := Lookup("moe-8x7b")
	if g.ActiveParams() >= g.TotalParams() {
		t.Fatalf("MoE active params %g should be well below total %g", g.ActiveParams(), g.TotalParams())
	}
	if got, want := g.ActiveParams(), 12.88e9; math.Abs(got-want) > 1e6 {
		t.Fatalf("ActiveParams() = %g, want %g", got, want)
	}
}

func TestKVBytesPerTokenKnownValueFor8BClass(t *testing.T) {
	// 2 (K,V) × 32 layers × 8 kv-heads × 128 head-dim × 2 bytes = 128 KiB —
	// the widely quoted per-token cache cost of the 8B GQA class at f16.
	g, _ := Lookup("8b")
	if got := g.KVBytesPerToken(2.0); got != 131072 {
		t.Fatalf("KVBytesPerToken(f16) = %g, want 131072", got)
	}
}

func TestKVBytesPerTokenScaleLinearlyWithElementWidth(t *testing.T) {
	g, _ := Lookup("8b")
	if g.KVBytesPerToken(1.0)*2 != g.KVBytesPerToken(2.0) {
		t.Fatal("KV bytes should scale linearly with element width")
	}
}

func TestClassicMHACachesCostMoreThanGQA(t *testing.T) {
	// The 7B MHA generation keeps 32 KV heads vs the 8B class's 8 — its
	// cache must be ~4x heavier per token despite the smaller model.
	mha, _ := Lookup("7b")
	gqa, _ := Lookup("8b")
	if mha.KVBytesPerToken(2) <= gqa.KVBytesPerToken(2) {
		t.Fatalf("7b (MHA) KV/token %g should exceed 8b (GQA) %g",
			mha.KVBytesPerToken(2), gqa.KVBytesPerToken(2))
	}
}

func TestTiedEmbeddingsReduceDerivedParams(t *testing.T) {
	g, _ := Lookup("1b")
	untied := g
	untied.Tied = false
	diff := untied.DerivedParams() - g.DerivedParams()
	want := float64(g.Vocab) * float64(g.DModel) // one extra output head
	if math.Abs(diff-want) > 1 {
		t.Fatalf("untying should add vocab×d_model = %g params, added %g", want, diff)
	}
}

func TestLookupCaseInsensitiveAndUnknownListsPresets(t *testing.T) {
	for _, in := range []string{"8b", "8B", " 8b "} {
		if _, err := Lookup(in); err != nil {
			t.Errorf("Lookup(%q) should resolve, got %v", in, err)
		}
	}
	_, err := Lookup("9000b")
	if err == nil {
		t.Fatal("Lookup(9000b) should fail")
	}
	if !strings.Contains(err.Error(), "70b") {
		t.Fatalf("error should list preset names, got: %v", err)
	}
}

func TestAllSortedByParamsAscending(t *testing.T) {
	all := All()
	for i := 1; i < len(all); i++ {
		if all[i].ParamsB < all[i-1].ParamsB {
			t.Fatalf("All() not sorted: %s (%.2fB) after %s (%.2fB)",
				all[i].Name, all[i].ParamsB, all[i-1].Name, all[i-1].ParamsB)
		}
	}
	if len(all) < 10 {
		t.Fatalf("expected at least 10 presets, got %d", len(all))
	}
	names := Names()
	if len(names) != len(all) {
		t.Fatalf("Names() length %d != All() length %d", len(names), len(all))
	}
	for i := range all {
		if names[i] != all[i].Name {
			t.Fatalf("Names()[%d] = %s, want %s", i, names[i], all[i].Name)
		}
	}
}

func TestEveryPresetValidates(t *testing.T) {
	for _, g := range All() {
		if err := g.Validate(); err != nil {
			t.Errorf("preset %s failed validation: %v", g.Name, err)
		}
	}
}

func TestValidateRejectsBrokenGeometries(t *testing.T) {
	base, _ := Lookup("8b")
	cases := []struct {
		name   string
		mutate func(*Geometry)
	}{
		{"zero layers", func(g *Geometry) { g.Layers = 0 }},
		{"zero params", func(g *Geometry) { g.ParamsB = 0 }},
		{"kv-heads above heads", func(g *Geometry) { g.KVHeads = 64 }},
		{"heads not multiple of kv-heads", func(g *Geometry) { g.KVHeads = 7 }},
		{"active above total", func(g *Geometry) { g.ActiveParamsB = 99 }},
		{"half-specified MoE", func(g *Geometry) { g.Experts = 8 }},
		{"active experts above experts", func(g *Geometry) { g.Experts = 4; g.ActiveExperts = 8 }},
	}
	for _, tc := range cases {
		g := base
		tc.mutate(&g)
		if err := g.Validate(); err == nil {
			t.Errorf("%s: Validate() should fail", tc.name)
		}
	}
}

func TestValidateRequiresHeadDimWhenIndivisible(t *testing.T) {
	g, _ := Lookup("8b")
	g.Heads = 30 // 4096 % 30 != 0, and kv-heads 8 no longer divides 30 either
	g.KVHeads = 30
	if err := g.Validate(); err == nil {
		t.Fatal("indivisible d_model without explicit head-dim should fail validation")
	}
	g.HeadDim = 128
	if err := g.Validate(); err != nil {
		t.Fatalf("explicit head-dim should make it valid, got %v", err)
	}
}
