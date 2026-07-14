// Package model describes transformer geometry — the handful of integers
// that, together with a quantization scheme, fully determine memory
// footprint and per-token traffic. Presets are generic size classes with
// real-world geometry (GQA ratios, FFN widths, vocab sizes as actually
// shipped by open dense and MoE models); every dense preset's declared
// parameter count is cross-checked against the count derived from its own
// geometry in the test suite.
package model

import (
	"fmt"
	"sort"
	"strings"
)

// Geometry is the shape of a decoder-only transformer.
type Geometry struct {
	Name          string
	ParamsB       float64 // declared total parameters, in billions
	ActiveParamsB float64 // parameters touched per token; 0 means dense (= ParamsB)
	Layers        int
	DModel        int
	Heads         int // query heads
	KVHeads       int // key/value heads (GQA); == Heads means classic MHA
	HeadDim       int // 0 means DModel / Heads
	FFN           int // feed-forward hidden width (per expert for MoE)
	Vocab         int
	Experts       int  // 0 for dense
	ActiveExperts int  // 0 for dense
	Tied          bool // input embedding shared with the output head
	MaxContext    int  // trained context window (informational)
	Note          string
}

// HeadDimResolved returns the per-head dimension, defaulting to DModel/Heads.
func (g Geometry) HeadDimResolved() int {
	if g.HeadDim > 0 {
		return g.HeadDim
	}
	return g.DModel / g.Heads
}

// TotalParams returns the declared total parameter count.
func (g Geometry) TotalParams() float64 { return g.ParamsB * 1e9 }

// ActiveParams returns the parameters read/multiplied per generated token.
// For dense models this equals TotalParams; for MoE it is the routed subset.
func (g Geometry) ActiveParams() float64 {
	if g.ActiveParamsB > 0 {
		return g.ActiveParamsB * 1e9
	}
	return g.TotalParams()
}

// MoE reports whether the model routes across experts.
func (g Geometry) MoE() bool { return g.Experts > 0 }

// KVBytesPerToken returns the KV-cache cost of one context position:
// K and V, per layer, per KV head, per head dimension, at the given
// element width. This is the number that decides long-context feasibility.
func (g Geometry) KVBytesPerToken(bytesPerElem float64) float64 {
	return 2 * float64(g.Layers) * float64(g.KVHeads) * float64(g.HeadDimResolved()) * bytesPerElem
}

// DerivedParams computes the parameter count implied by the geometry alone:
// embeddings (doubled unless tied), attention projections sized by explicit
// head dimension, gated FFN (3 matrices), per-layer norms, and — for MoE —
// all experts plus the router. Used to keep presets honest.
func (g Geometry) DerivedParams() float64 { return g.derived(false) }

// DerivedActiveParams is DerivedParams restricted to the experts actually
// routed per token. Equal to DerivedParams for dense models.
func (g Geometry) DerivedActiveParams() float64 { return g.derived(true) }

func (g Geometry) derived(activeOnly bool) float64 {
	d := float64(g.DModel)
	hd := float64(g.HeadDimResolved())
	qWidth := float64(g.Heads) * hd
	kvWidth := float64(g.KVHeads) * hd
	attn := d*qWidth + qWidth*d + 2*d*kvWidth // Q, O, K, V projections
	expertCount := 1.0
	router := 0.0
	if g.MoE() {
		expertCount = float64(g.Experts)
		if activeOnly {
			expertCount = float64(g.ActiveExperts)
		}
		router = d * float64(g.Experts)
	}
	ffn := 3 * d * float64(g.FFN) * expertCount // gated FFN: up, gate, down
	norms := 2 * d
	perLayer := attn + ffn + norms + router
	embed := float64(g.Vocab) * d
	if !g.Tied {
		embed *= 2 // separate output head
	}
	return float64(g.Layers)*perLayer + embed + d // + final norm
}

// Validate rejects geometries that cannot describe a real transformer.
func (g Geometry) Validate() error {
	switch {
	case g.ParamsB <= 0:
		return fmt.Errorf("model %q: params must be > 0", g.Name)
	case g.Layers <= 0 || g.DModel <= 0 || g.Heads <= 0 || g.FFN <= 0 || g.Vocab <= 0:
		return fmt.Errorf("model %q: layers, d-model, heads, ffn and vocab must all be > 0", g.Name)
	case g.KVHeads <= 0 || g.KVHeads > g.Heads:
		return fmt.Errorf("model %q: kv-heads must be in 1..heads", g.Name)
	case g.Heads%g.KVHeads != 0:
		return fmt.Errorf("model %q: heads (%d) must be a multiple of kv-heads (%d)", g.Name, g.Heads, g.KVHeads)
	case g.HeadDim == 0 && g.DModel%g.Heads != 0:
		return fmt.Errorf("model %q: d-model (%d) not divisible by heads (%d); set head-dim explicitly", g.Name, g.DModel, g.Heads)
	case g.ActiveParamsB > g.ParamsB:
		return fmt.Errorf("model %q: active params exceed total params", g.Name)
	case (g.Experts > 0) != (g.ActiveExperts > 0):
		return fmt.Errorf("model %q: experts and active-experts must be set together", g.Name)
	case g.Experts > 0 && g.ActiveExperts > g.Experts:
		return fmt.Errorf("model %q: active-experts exceed experts", g.Name)
	}
	return nil
}

// presets are generic size classes carrying the geometry those classes
// actually ship with. Declared params are the published counts; the test
// suite asserts DerivedParams matches them within 2%.
var presets = []Geometry{
	{Name: "1b", ParamsB: 1.24, Layers: 16, DModel: 2048, Heads: 32, KVHeads: 8, HeadDim: 64, FFN: 8192, Vocab: 128256, Tied: true, MaxContext: 131072, Note: "edge-class dense model, tied embeddings"},
	{Name: "3b", ParamsB: 3.21, Layers: 28, DModel: 3072, Heads: 24, KVHeads: 8, HeadDim: 128, FFN: 8192, Vocab: 128256, Tied: true, MaxContext: 131072, Note: "small dense model, tied embeddings"},
	{Name: "7b", ParamsB: 6.74, Layers: 32, DModel: 4096, Heads: 32, KVHeads: 32, FFN: 11008, Vocab: 32000, MaxContext: 4096, Note: "classic MHA generation; heaviest KV cache per token"},
	{Name: "8b", ParamsB: 8.03, Layers: 32, DModel: 4096, Heads: 32, KVHeads: 8, FFN: 14336, Vocab: 128256, MaxContext: 131072, Note: "modern GQA workhorse"},
	{Name: "13b", ParamsB: 13.02, Layers: 40, DModel: 5120, Heads: 40, KVHeads: 40, FFN: 13824, Vocab: 32000, MaxContext: 4096, Note: "classic MHA generation, mid size"},
	{Name: "14b", ParamsB: 14.77, Layers: 48, DModel: 5120, Heads: 40, KVHeads: 8, HeadDim: 128, FFN: 13824, Vocab: 152064, MaxContext: 131072, Note: "deep GQA mid-size class"},
	{Name: "24b", ParamsB: 23.57, Layers: 40, DModel: 5120, Heads: 32, KVHeads: 8, HeadDim: 128, FFN: 32768, Vocab: 131072, MaxContext: 32768, Note: "wide-FFN efficient mid-size class"},
	{Name: "32b", ParamsB: 32.76, Layers: 64, DModel: 5120, Heads: 40, KVHeads: 8, HeadDim: 128, FFN: 27648, Vocab: 152064, MaxContext: 131072, Note: "large single-GPU class"},
	{Name: "70b", ParamsB: 70.55, Layers: 80, DModel: 8192, Heads: 64, KVHeads: 8, FFN: 28672, Vocab: 128256, MaxContext: 131072, Note: "flagship dense class"},
	{Name: "moe-8x7b", ParamsB: 46.70, ActiveParamsB: 12.88, Layers: 32, DModel: 4096, Heads: 32, KVHeads: 8, FFN: 14336, Vocab: 32000, Experts: 8, ActiveExperts: 2, MaxContext: 32768, Note: "sparse MoE: 47B footprint, 13B per-token traffic"},
	{Name: "moe-30b-a3b", ParamsB: 30.53, ActiveParamsB: 3.34, Layers: 48, DModel: 2048, Heads: 32, KVHeads: 4, HeadDim: 128, FFN: 768, Vocab: 151936, Experts: 128, ActiveExperts: 8, MaxContext: 32768, Note: "fine-grained MoE: 30B footprint, 3B per-token traffic"},
}

// Lookup resolves a preset by name (case-insensitive).
func Lookup(name string) (Geometry, error) {
	key := strings.ToLower(strings.TrimSpace(name))
	for _, g := range presets {
		if g.Name == key {
			return g, nil
		}
	}
	return Geometry{}, fmt.Errorf("unknown model %q (known: %s)", name, strings.Join(Names(), ", "))
}

// All returns the presets ordered by total params ascending, name as
// tie-break — the order every listing uses.
func All() []Geometry {
	out := make([]Geometry, len(presets))
	copy(out, presets)
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].ParamsB != out[j].ParamsB {
			return out[i].ParamsB < out[j].ParamsB
		}
		return out[i].Name < out[j].Name
	})
	return out
}

// Names lists preset names in All() order.
func Names() []string {
	all := All()
	names := make([]string, len(all))
	for i, g := range all {
		names[i] = g.Name
	}
	return names
}
