// Package roofline turns a device, a model geometry and a quantization
// choice into closed-form tokens-per-second bounds. The model is the
// classic roofline argument specialized to autoregressive transformers:
//
//   - Decode (one token at a time) must stream every active weight byte and
//     the whole KV cache past the compute units for each token, so it is
//     bounded by memory bandwidth; it is also bounded by the FLOPs the chip
//     can retire. The achievable rate is the minimum of the two, and on
//     every realistic device the bandwidth bound binds.
//   - Prefill processes the whole prompt as one batch, amortizing weight
//     traffic across positions, so it is (almost always) compute-bound.
//
// Every quantity is derived, none is measured — that is the point: you can
// run this for hardware you do not own. Efficiency bands translate
// theoretical peaks into the fraction real inference stacks reach.
// Derivations, calibration and limitations live in docs/method.md.
package roofline

import (
	"fmt"
	"math"

	"github.com/JaydenCJ/inferest/internal/device"
	"github.com/JaydenCJ/inferest/internal/model"
	"github.com/JaydenCJ/inferest/internal/quant"
)

// Band holds a conservative / expected / optimistic triple. Bands express
// honest uncertainty about kernel quality without pretending to benchmark.
type Band struct {
	Conservative float64
	Expected     float64
	Optimistic   float64
}

// scale multiplies a theoretical peak by each efficiency in the band.
func (b Band) scale(peak float64) Band {
	return Band{peak * b.Conservative, peak * b.Expected, peak * b.Optimistic}
}

// minBand takes the element-wise minimum of two bands (the roofline "min of
// bounds" applied per efficiency scenario).
func minBand(a, b Band) Band {
	return Band{
		math.Min(a.Conservative, b.Conservative),
		math.Min(a.Expected, b.Expected),
		math.Min(a.Optimistic, b.Optimistic),
	}
}

// Default efficiency bands. Sources and calibration in docs/method.md:
// measured decode throughput on well-optimized stacks lands at 55–85% of
// spec-sheet bandwidth; prefill model FLOPs utilization lands at 25–55%.
var (
	DefaultBandwidthEff = Band{0.55, 0.70, 0.85}
	DefaultMFU          = Band{0.25, 0.40, 0.55}
)

// Memory-overhead model beyond weights and KV cache (docs/method.md §5):
// a fixed runtime allocation, a small fraction of weights for fragmentation
// and dequant scratch, and a per-context-token compute buffer.
const (
	fixedOverheadBytes    = 256 << 20 // runtime, logits, tokenizer tables
	weightOverheadFrac    = 0.02      // fragmentation + scratch, ~2% of weights
	computeBufBytesPerDim = 8.0       // activation buffers: 8 B × d_model × context
)

// Inputs is a fully-specified estimation request.
type Inputs struct {
	Device  device.Device
	Model   model.Geometry
	Weights quant.Scheme
	KVCache quant.KVScheme
	Context int // context window to plan for (prompt + generation)
	Prompt  int // prompt length used for prefill / time-to-first-token

	// BandwidthEff / MFU, when > 0, collapse the corresponding band to a
	// single user-chosen efficiency (conservative = expected = optimistic).
	BandwidthEff float64
	MFU          float64
}

func (in Inputs) bandwidthBand() Band {
	if in.BandwidthEff > 0 {
		return Band{in.BandwidthEff, in.BandwidthEff, in.BandwidthEff}
	}
	return DefaultBandwidthEff
}

func (in Inputs) mfuBand() Band {
	if in.MFU > 0 {
		return Band{in.MFU, in.MFU, in.MFU}
	}
	return DefaultMFU
}

// Validate rejects requests the math cannot answer.
func (in Inputs) Validate() error {
	if err := in.Device.Validate(); err != nil {
		return err
	}
	if err := in.Model.Validate(); err != nil {
		return err
	}
	if in.Context < 0 {
		return fmt.Errorf("context must be >= 0, got %d", in.Context)
	}
	if in.Prompt < 1 {
		return fmt.Errorf("prompt must be >= 1 token, got %d", in.Prompt)
	}
	if in.Prompt > in.Context {
		return fmt.Errorf("prompt (%d) cannot exceed context (%d)", in.Prompt, in.Context)
	}
	if in.BandwidthEff < 0 || in.BandwidthEff > 1 {
		return fmt.Errorf("bandwidth efficiency must be in (0,1], got %g", in.BandwidthEff)
	}
	if in.MFU < 0 || in.MFU > 1 {
		return fmt.Errorf("mfu must be in (0,1], got %g", in.MFU)
	}
	return nil
}

// Memory is the footprint breakdown at the requested context.
type Memory struct {
	WeightBytes     float64
	KVBytesPerToken float64
	KVBytes         float64 // at Inputs.Context
	OverheadBytes   float64 // fixed + weight fraction + compute buffers
	TotalBytes      float64
	CapacityBytes   float64 // 0 = unknown
	Known           bool    // capacity was provided
	Fits            bool    // meaningful only when Known
	UsedFraction    float64 // meaningful only when Known
	MaxContext      int     // largest context that fits; -1 when capacity unknown
}

// DecodePoint is the decode-speed estimate at one context position.
type DecodePoint struct {
	ContextTokens  int
	BytesPerToken  float64
	FlopsPerToken  float64
	BandwidthBound Band // t/s if only bandwidth mattered
	ComputeBound   Band // t/s if only compute mattered
	TPS            Band // min of the two, per scenario
}

// Decode aggregates decode-speed estimates across context fill levels.
type Decode struct {
	Points []DecodePoint // at 0, Context/2 and Context tokens of history
	// BandwidthLimited reports whether, at full context and expected
	// efficiency, bandwidth (not compute) is the binding constraint.
	BandwidthLimited bool
	// ComputeHeadroom is computeBound/bandwidthBound at full context and
	// expected efficiency: how many times faster the chip could go if
	// memory were infinite. >1 means bandwidth-bound.
	ComputeHeadroom float64
}

// Prefill is the prompt-processing estimate.
type Prefill struct {
	PromptTokens   int
	FlopsPerToken  float64
	BytesPerToken  float64
	ComputeBound   Band
	BandwidthBound Band
	TPS            Band // min of the two, per scenario
	TTFTSeconds    Band // prompt / TPS (note: conservative TPS ⇒ optimistic-largest TTFT)
	ComputeLimited bool
}

// Estimate is the full closed-form result.
type Estimate struct {
	In      Inputs
	Memory  Memory
	Decode  Decode
	Prefill Prefill
}

// New computes the estimate. It is a pure function of Inputs.
func New(in Inputs) (Estimate, error) {
	if err := in.Validate(); err != nil {
		return Estimate{}, err
	}
	est := Estimate{In: in}
	est.Memory = memoryAt(in)
	est.Decode = decode(in)
	est.Prefill = prefill(in)
	return est, nil
}

// memoryAt builds the footprint breakdown and solves for max context.
func memoryAt(in Inputs) Memory {
	m := Memory{
		WeightBytes:     in.Weights.BytesForParams(in.Model.TotalParams()),
		KVBytesPerToken: in.Model.KVBytesPerToken(in.KVCache.BytesPerElem),
	}
	m.KVBytes = m.KVBytesPerToken * float64(in.Context)
	computeBuf := computeBufBytesPerDim * float64(in.Model.DModel) * float64(in.Context)
	m.OverheadBytes = fixedOverheadBytes + weightOverheadFrac*m.WeightBytes + computeBuf
	m.TotalBytes = m.WeightBytes + m.KVBytes + m.OverheadBytes

	m.CapacityBytes = in.Device.MemoryBytes()
	m.Known = m.CapacityBytes > 0
	if !m.Known {
		m.MaxContext = -1
		return m
	}
	m.Fits = m.TotalBytes <= m.CapacityBytes
	m.UsedFraction = m.TotalBytes / m.CapacityBytes

	// Solve capacity >= weights·(1+frac) + fixed + ctx·(kv/token + 8·d_model)
	// for the largest integer ctx.
	perToken := m.KVBytesPerToken + computeBufBytesPerDim*float64(in.Model.DModel)
	budget := m.CapacityBytes - m.WeightBytes*(1+weightOverheadFrac) - fixedOverheadBytes
	if budget < 0 {
		m.MaxContext = 0 // weights alone do not fit
	} else {
		m.MaxContext = int(budget / perToken)
	}
	return m
}

// decodeBytesPerToken: every active weight byte plus the whole KV cache
// must be read once per generated token.
func decodeBytesPerToken(in Inputs, contextTokens int) float64 {
	activeWeights := in.Weights.BytesForParams(in.Model.ActiveParams())
	kv := in.Model.KVBytesPerToken(in.KVCache.BytesPerElem) * float64(contextTokens)
	return activeWeights + kv
}

// decodeFlopsPerToken: 2 FLOPs per active weight (multiply-accumulate) plus
// attention score/value FLOPs, which grow linearly with context.
func decodeFlopsPerToken(in Inputs, contextTokens int) float64 {
	attnWidth := float64(in.Model.Heads) * float64(in.Model.HeadDimResolved())
	attn := 4 * float64(in.Model.Layers) * attnWidth * float64(contextTokens)
	return 2*in.Model.ActiveParams() + attn
}

func decode(in Inputs) Decode {
	positions := []int{0, in.Context / 2, in.Context}
	var d Decode
	seen := map[int]bool{}
	for _, pos := range positions {
		if seen[pos] {
			continue
		}
		seen[pos] = true
		d.Points = append(d.Points, decodeAt(in, pos))
	}
	last := d.Points[len(d.Points)-1]
	d.BandwidthLimited = last.BandwidthBound.Expected <= last.ComputeBound.Expected
	d.ComputeHeadroom = last.ComputeBound.Expected / last.BandwidthBound.Expected
	return d
}

func decodeAt(in Inputs, contextTokens int) DecodePoint {
	p := DecodePoint{
		ContextTokens: contextTokens,
		BytesPerToken: decodeBytesPerToken(in, contextTokens),
		FlopsPerToken: decodeFlopsPerToken(in, contextTokens),
	}
	p.BandwidthBound = in.bandwidthBand().scale(in.Device.BandwidthBytesPerSec() / p.BytesPerToken)
	p.ComputeBound = in.mfuBand().scale(in.Device.FLOPS() / p.FlopsPerToken)
	p.TPS = minBand(p.BandwidthBound, p.ComputeBound)
	return p
}

func prefill(in Inputs) Prefill {
	p := Prefill{PromptTokens: in.Prompt}
	n := float64(in.Prompt)
	attnWidth := float64(in.Model.Heads) * float64(in.Model.HeadDimResolved())
	// Per-token FLOPs averaged over the prompt: weight FLOPs are constant,
	// causal attention contributes 2·L·width·P per token (½ · 4·L·width·P).
	p.FlopsPerToken = 2*in.Model.ActiveParams() + 2*float64(in.Model.Layers)*attnWidth*n
	// Per-token bytes: weights are read once for the whole batched pass and
	// amortize across the prompt; each position also writes its KV entry.
	activeWeights := in.Weights.BytesForParams(in.Model.ActiveParams())
	p.BytesPerToken = activeWeights/n + in.Model.KVBytesPerToken(in.KVCache.BytesPerElem)

	p.ComputeBound = in.mfuBand().scale(in.Device.FLOPS() / p.FlopsPerToken)
	p.BandwidthBound = in.bandwidthBand().scale(in.Device.BandwidthBytesPerSec() / p.BytesPerToken)
	p.TPS = minBand(p.ComputeBound, p.BandwidthBound)
	p.ComputeLimited = p.ComputeBound.Expected <= p.BandwidthBound.Expected
	p.TTFTSeconds = Band{
		Conservative: n / p.TPS.Conservative,
		Expected:     n / p.TPS.Expected,
		Optimistic:   n / p.TPS.Optimistic,
	}
	return p
}
