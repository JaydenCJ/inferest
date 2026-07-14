package render

import (
	"encoding/json"

	"github.com/JaydenCJ/inferest/internal/device"
	"github.com/JaydenCJ/inferest/internal/model"
	"github.com/JaydenCJ/inferest/internal/quant"
	"github.com/JaydenCJ/inferest/internal/roofline"
)

// SchemaVersion identifies the JSON envelope layout. Bump on breaking
// changes so downstream scripts can pin what they parse.
const SchemaVersion = 1

type envelope struct {
	Tool          string `json:"tool"`
	SchemaVersion int    `json:"schema_version"`
	Command       string `json:"command"`
}

func newEnvelope(command string) envelope {
	return envelope{Tool: "inferest", SchemaVersion: SchemaVersion, Command: command}
}

type bandJSON struct {
	Conservative float64 `json:"conservative"`
	Expected     float64 `json:"expected"`
	Optimistic   float64 `json:"optimistic"`
}

func toBand(b roofline.Band) bandJSON {
	return bandJSON{b.Conservative, b.Expected, b.Optimistic}
}

type deviceJSON struct {
	Name         string  `json:"name"`
	Kind         string  `json:"kind"`
	MemoryGiB    float64 `json:"memory_gib"`
	BandwidthGBs float64 `json:"bandwidth_gb_s"`
	TFLOPSFP16   float64 `json:"tflops_fp16"`
	Note         string  `json:"note,omitempty"`
}

func toDevice(d device.Device) deviceJSON {
	return deviceJSON{d.Name, d.Kind, d.MemoryGiB, d.BandwidthGBs, d.TFLOPSFP16, d.Note}
}

type modelJSON struct {
	Name          string  `json:"name"`
	ParamsB       float64 `json:"params_b"`
	ActiveParamsB float64 `json:"active_params_b"`
	Layers        int     `json:"layers"`
	DModel        int     `json:"d_model"`
	Heads         int     `json:"heads"`
	KVHeads       int     `json:"kv_heads"`
	HeadDim       int     `json:"head_dim"`
	FFN           int     `json:"ffn"`
	Vocab         int     `json:"vocab"`
	Experts       int     `json:"experts,omitempty"`
	ActiveExperts int     `json:"active_experts,omitempty"`
	Tied          bool    `json:"tied_embeddings"`
	MaxContext    int     `json:"max_context,omitempty"`
	Note          string  `json:"note,omitempty"`
}

func toModel(g model.Geometry) modelJSON {
	return modelJSON{
		Name: g.Name, ParamsB: g.ParamsB, ActiveParamsB: g.ActiveParams() / 1e9,
		Layers: g.Layers, DModel: g.DModel, Heads: g.Heads, KVHeads: g.KVHeads,
		HeadDim: g.HeadDimResolved(), FFN: g.FFN, Vocab: g.Vocab,
		Experts: g.Experts, ActiveExperts: g.ActiveExperts, Tied: g.Tied,
		MaxContext: g.MaxContext, Note: g.Note,
	}
}

type memoryJSON struct {
	WeightBytes     float64 `json:"weight_bytes"`
	KVBytesPerToken float64 `json:"kv_bytes_per_token"`
	KVBytes         float64 `json:"kv_bytes"`
	OverheadBytes   float64 `json:"overhead_bytes"`
	TotalBytes      float64 `json:"total_bytes"`
	CapacityBytes   float64 `json:"capacity_bytes"`
	CapacityKnown   bool    `json:"capacity_known"`
	Fits            bool    `json:"fits"`
	UsedFraction    float64 `json:"used_fraction"`
	MaxContext      int     `json:"max_context"`
}

func toMemory(m roofline.Memory) memoryJSON {
	return memoryJSON{m.WeightBytes, m.KVBytesPerToken, m.KVBytes, m.OverheadBytes,
		m.TotalBytes, m.CapacityBytes, m.Known, m.Fits, m.UsedFraction, m.MaxContext}
}

type decodePointJSON struct {
	ContextTokens     int      `json:"context_tokens"`
	BytesPerToken     float64  `json:"bytes_per_token"`
	FlopsPerToken     float64  `json:"flops_per_token"`
	BandwidthBoundTPS bandJSON `json:"bandwidth_bound_tps"`
	ComputeBoundTPS   bandJSON `json:"compute_bound_tps"`
	TPS               bandJSON `json:"tps"`
}

type decodeJSON struct {
	Points           []decodePointJSON `json:"points"`
	BandwidthLimited bool              `json:"bandwidth_limited"`
	ComputeHeadroom  float64           `json:"compute_headroom"`
}

type prefillJSON struct {
	PromptTokens   int      `json:"prompt_tokens"`
	FlopsPerToken  float64  `json:"flops_per_token"`
	BytesPerToken  float64  `json:"bytes_per_token"`
	TPS            bandJSON `json:"tps"`
	TTFTSeconds    bandJSON `json:"ttft_seconds"`
	ComputeLimited bool     `json:"compute_limited"`
}

type inputsJSON struct {
	Device       deviceJSON `json:"device"`
	Model        modelJSON  `json:"model"`
	Quant        string     `json:"quant"`
	BitsPerWt    float64    `json:"bits_per_weight"`
	KVCache      string     `json:"kv_cache"`
	KVBytesElem  float64    `json:"kv_bytes_per_element"`
	Context      int        `json:"context"`
	Prompt       int        `json:"prompt"`
	BandwidthEff bandJSON   `json:"bandwidth_efficiency"`
	MFU          bandJSON   `json:"mfu"`
}

func toInputs(est roofline.Estimate) inputsJSON {
	in := est.In
	bw, mfu := roofline.DefaultBandwidthEff, roofline.DefaultMFU
	if in.BandwidthEff > 0 {
		bw = roofline.Band{Conservative: in.BandwidthEff, Expected: in.BandwidthEff, Optimistic: in.BandwidthEff}
	}
	if in.MFU > 0 {
		mfu = roofline.Band{Conservative: in.MFU, Expected: in.MFU, Optimistic: in.MFU}
	}
	return inputsJSON{
		Device: toDevice(in.Device), Model: toModel(in.Model),
		Quant: in.Weights.Name, BitsPerWt: in.Weights.BitsPerWeight,
		KVCache: in.KVCache.Name, KVBytesElem: in.KVCache.BytesPerElem,
		Context: in.Context, Prompt: in.Prompt,
		BandwidthEff: toBand(bw), MFU: toBand(mfu),
	}
}

func toDecode(d roofline.Decode) decodeJSON {
	out := decodeJSON{BandwidthLimited: d.BandwidthLimited, ComputeHeadroom: d.ComputeHeadroom}
	for _, p := range d.Points {
		out.Points = append(out.Points, decodePointJSON{
			ContextTokens: p.ContextTokens, BytesPerToken: p.BytesPerToken,
			FlopsPerToken: p.FlopsPerToken, BandwidthBoundTPS: toBand(p.BandwidthBound),
			ComputeBoundTPS: toBand(p.ComputeBound), TPS: toBand(p.TPS),
		})
	}
	return out
}

func toPrefill(p roofline.Prefill) prefillJSON {
	return prefillJSON{
		PromptTokens: p.PromptTokens, FlopsPerToken: p.FlopsPerToken,
		BytesPerToken: p.BytesPerToken, TPS: toBand(p.TPS),
		TTFTSeconds: toBand(p.TTFTSeconds), ComputeLimited: p.ComputeLimited,
	}
}

// EstimateJSON renders the full estimate as indented JSON.
func EstimateJSON(est roofline.Estimate) ([]byte, error) {
	doc := struct {
		envelope
		Inputs  inputsJSON  `json:"inputs"`
		Memory  memoryJSON  `json:"memory"`
		Decode  decodeJSON  `json:"decode"`
		Prefill prefillJSON `json:"prefill"`
	}{newEnvelope("estimate"), toInputs(est), toMemory(est.Memory), toDecode(est.Decode), toPrefill(est.Prefill)}
	return json.MarshalIndent(doc, "", "  ")
}

// CompareJSON renders one result row per device.
func CompareJSON(ests []roofline.Estimate) ([]byte, error) {
	type row struct {
		Device       deviceJSON `json:"device"`
		Fits         bool       `json:"fits"`
		Known        bool       `json:"capacity_known"`
		UsedFraction float64    `json:"used_fraction"`
		MaxContext   int        `json:"max_context"`
		DecodeTPS    bandJSON   `json:"decode_tps_at_context"`
		PrefillTPS   bandJSON   `json:"prefill_tps"`
		TTFTSeconds  bandJSON   `json:"ttft_seconds"`
	}
	doc := struct {
		envelope
		Model   modelJSON `json:"model"`
		Quant   string    `json:"quant"`
		KVCache string    `json:"kv_cache"`
		Context int       `json:"context"`
		Prompt  int       `json:"prompt"`
		Results []row     `json:"results"`
	}{envelope: newEnvelope("compare")}
	if len(ests) > 0 {
		in := ests[0].In
		doc.Model, doc.Quant, doc.KVCache = toModel(in.Model), in.Weights.Name, in.KVCache.Name
		doc.Context, doc.Prompt = in.Context, in.Prompt
	}
	for _, est := range ests {
		last := est.Decode.Points[len(est.Decode.Points)-1]
		doc.Results = append(doc.Results, row{
			Device: toDevice(est.In.Device), Fits: est.Memory.Fits, Known: est.Memory.Known,
			UsedFraction: est.Memory.UsedFraction, MaxContext: est.Memory.MaxContext,
			DecodeTPS: toBand(last.TPS), PrefillTPS: toBand(est.Prefill.TPS),
			TTFTSeconds: toBand(est.Prefill.TTFTSeconds),
		})
	}
	return json.MarshalIndent(doc, "", "  ")
}

// FitJSON renders the memory verdict.
func FitJSON(est roofline.Estimate, bestQuant *quant.Scheme) ([]byte, error) {
	best := ""
	if bestQuant != nil {
		best = bestQuant.Name
	}
	doc := struct {
		envelope
		Inputs             inputsJSON `json:"inputs"`
		Memory             memoryJSON `json:"memory"`
		WidestQuantThatFit string     `json:"widest_quant_that_fits,omitempty"`
	}{newEnvelope("fit"), toInputs(est), toMemory(est.Memory), best}
	return json.MarshalIndent(doc, "", "  ")
}

// DevicesJSON renders the device preset list.
func DevicesJSON(devs []device.Device) ([]byte, error) {
	rows := make([]deviceJSON, len(devs))
	for i, d := range devs {
		rows[i] = toDevice(d)
	}
	doc := struct {
		envelope
		Devices []deviceJSON `json:"devices"`
	}{newEnvelope("devices"), rows}
	return json.MarshalIndent(doc, "", "  ")
}

// ModelsJSON renders the model preset list.
func ModelsJSON(models []model.Geometry) ([]byte, error) {
	rows := make([]modelJSON, len(models))
	for i, g := range models {
		rows[i] = toModel(g)
	}
	doc := struct {
		envelope
		Models []modelJSON `json:"models"`
	}{newEnvelope("models"), rows}
	return json.MarshalIndent(doc, "", "  ")
}

// QuantsJSON renders both quantization tables.
func QuantsJSON(ws []quant.Scheme, kvs []quant.KVScheme) ([]byte, error) {
	type wRow struct {
		Name          string  `json:"name"`
		BitsPerWeight float64 `json:"bits_per_weight"`
		Note          string  `json:"note"`
	}
	type kvRow struct {
		Name         string  `json:"name"`
		BytesPerElem float64 `json:"bytes_per_element"`
		Note         string  `json:"note"`
	}
	doc := struct {
		envelope
		Weights []wRow  `json:"weights"`
		KVCache []kvRow `json:"kv_cache"`
	}{envelope: newEnvelope("quants")}
	for _, s := range ws {
		doc.Weights = append(doc.Weights, wRow{s.Name, s.BitsPerWeight, s.Note})
	}
	for _, s := range kvs {
		doc.KVCache = append(doc.KVCache, kvRow{s.Name, s.BytesPerElem, s.Note})
	}
	return json.MarshalIndent(doc, "", "  ")
}
