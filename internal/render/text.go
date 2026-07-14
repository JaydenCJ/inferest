package render

import (
	"fmt"
	"io"

	"github.com/JaydenCJ/inferest/internal/device"
	"github.com/JaydenCJ/inferest/internal/model"
	"github.com/JaydenCJ/inferest/internal/quant"
	"github.com/JaydenCJ/inferest/internal/roofline"
)

// deviceLine summarizes a device on one line.
func deviceLine(d device.Device) string {
	mem := "memory n/a"
	if d.MemoryGiB > 0 {
		mem = fmt.Sprintf("%.1f GiB", d.MemoryGiB)
	}
	return fmt.Sprintf("%s · %s · %s · %g GB/s · %g TFLOPS fp16", d.Name, d.Kind, mem, d.BandwidthGBs, d.TFLOPSFP16)
}

// modelLine summarizes a geometry on one line.
func modelLine(g model.Geometry) string {
	shape := fmt.Sprintf("%d layers · d_model %d · heads %d/kv %d", g.Layers, g.DModel, g.Heads, g.KVHeads)
	if g.MoE() {
		return fmt.Sprintf("%s · %.2fB total / %.2fB active · %d experts (%d active) · %s",
			g.Name, g.ParamsB, g.ActiveParamsB, g.Experts, g.ActiveExperts, shape)
	}
	return fmt.Sprintf("%s · %.2fB params · dense · %s", g.Name, g.ParamsB, shape)
}

// quantLine summarizes the quantization choice on one line.
func quantLine(q quant.Scheme, kv quant.KVScheme, kvPerToken float64) string {
	return fmt.Sprintf("%s · %.2f bits/weight effective · kv cache %s · %s per context token",
		q.Name, q.BitsPerWeight, kv.Name, fmtBytes(kvPerToken))
}

// memorySection writes the footprint breakdown shared by estimate and fit.
func memorySection(w io.Writer, est roofline.Estimate) {
	m := est.Memory
	fmt.Fprintf(w, "memory @ context %s\n", group(est.In.Context))
	fmt.Fprintf(w, "  weights     %12s\n", fmtBytes(m.WeightBytes))
	fmt.Fprintf(w, "  kv cache    %12s\n", fmtBytes(m.KVBytes))
	fmt.Fprintf(w, "  overhead    %12s\n", fmtBytes(m.OverheadBytes))
	if !m.Known {
		fmt.Fprintf(w, "  total       %12s / capacity unknown — pass --memory-gb for a fit verdict\n", fmtBytes(m.TotalBytes))
		return
	}
	verdict := "FITS"
	if !m.Fits {
		verdict = "DOES NOT FIT"
	}
	fmt.Fprintf(w, "  total       %12s / %s   %s   (%s of memory)\n",
		fmtBytes(m.TotalBytes), fmtBytes(m.CapacityBytes), verdict, fmtPct(m.UsedFraction))
	switch {
	case m.MaxContext == 0:
		fmt.Fprintf(w, "  max context  weights alone exceed capacity\n")
	case est.In.Model.MaxContext > 0 && m.MaxContext > est.In.Model.MaxContext:
		fmt.Fprintf(w, "  max context  ≈ %s tokens on this device (model trained to %s)\n",
			group(m.MaxContext), group(est.In.Model.MaxContext))
	default:
		fmt.Fprintf(w, "  max context  ≈ %s tokens on this device\n", group(m.MaxContext))
	}
}

// Estimate writes the full text report.
func Estimate(w io.Writer, est roofline.Estimate) {
	in := est.In
	fmt.Fprintf(w, "inferest estimate — %s @ %s on %s\n\n", in.Model.Name, in.Weights.Name, in.Device.Name)
	fmt.Fprintf(w, "device   %s\n", deviceLine(in.Device))
	fmt.Fprintf(w, "model    %s\n", modelLine(in.Model))
	fmt.Fprintf(w, "quant    %s\n\n", quantLine(in.Weights, in.KVCache, est.Memory.KVBytesPerToken))

	memorySection(w, est)

	fmt.Fprintf(w, "\ndecode speed, t/s (single stream)\n")
	fmt.Fprintf(w, "  %-18s %12s %12s %12s\n", "context", "conservative", "expected", "optimistic")
	for _, p := range est.Decode.Points {
		label := "empty"
		if p.ContextTokens > 0 {
			label = group(p.ContextTokens) + " tokens"
		}
		fmt.Fprintf(w, "  %-18s %12s %12s %12s\n", label,
			fmtTPS(p.TPS.Conservative), fmtTPS(p.TPS.Expected), fmtTPS(p.TPS.Optimistic))
	}
	if est.Decode.BandwidthLimited {
		fmt.Fprintf(w, "  bound: memory bandwidth (compute headroom %.1fx)\n", est.Decode.ComputeHeadroom)
	} else {
		fmt.Fprintf(w, "  bound: compute (bandwidth headroom %.1fx)\n", 1/est.Decode.ComputeHeadroom)
	}

	pf := est.Prefill
	fmt.Fprintf(w, "\nprefill (%s-token prompt)\n", group(pf.PromptTokens))
	fmt.Fprintf(w, "  speed   %s / %s / %s t/s   (conservative / expected / optimistic)\n",
		fmtTPS(pf.TPS.Conservative), fmtTPS(pf.TPS.Expected), fmtTPS(pf.TPS.Optimistic))
	fmt.Fprintf(w, "  ttft    %s / %s / %s\n",
		fmtSeconds(pf.TTFTSeconds.Conservative), fmtSeconds(pf.TTFTSeconds.Expected), fmtSeconds(pf.TTFTSeconds.Optimistic))
	if pf.ComputeLimited {
		fmt.Fprintf(w, "  bound: compute\n")
	} else {
		fmt.Fprintf(w, "  bound: memory bandwidth\n")
	}
}

// Compare writes one table row per device for the same model/quant/context.
func Compare(w io.Writer, ests []roofline.Estimate) {
	if len(ests) == 0 {
		return
	}
	in := ests[0].In
	fmt.Fprintf(w, "inferest compare — %s @ %s · context %s · prompt %s\n\n",
		in.Model.Name, in.Weights.Name, group(in.Context), group(in.Prompt))
	fmt.Fprintf(w, "%-22s %14s %10s %15s %10s %10s\n",
		"device", "fit", "decode", "range (c–o)", "prefill", "ttft")
	for _, est := range ests {
		m := est.Memory
		fit := "?"
		if m.Known {
			fit = fmtPct(m.UsedFraction)
			if !m.Fits {
				fit = "DOES NOT FIT"
			}
		}
		decodeCol, rangeCol, prefillCol, ttftCol := "—", "—", "—", "—"
		if !m.Known || m.Fits {
			last := est.Decode.Points[len(est.Decode.Points)-1]
			decodeCol = fmtTPS(last.TPS.Expected)
			rangeCol = fmtTPS(last.TPS.Conservative) + "–" + fmtTPS(last.TPS.Optimistic)
			prefillCol = fmtTPS(est.Prefill.TPS.Expected)
			ttftCol = fmtSeconds(est.Prefill.TTFTSeconds.Expected)
		}
		fmt.Fprintf(w, "%-22s %14s %10s %15s %10s %10s\n",
			est.In.Device.Name, fit, decodeCol, rangeCol, prefillCol, ttftCol)
	}
	fmt.Fprintf(w, "\ndecode/prefill/ttft are expected-efficiency figures at full context; — = does not fit\n")
}

// Fit writes the memory verdict. bestQuant, when non-nil, names the widest
// scheme that would fit at the same context (computed by the caller).
func Fit(w io.Writer, est roofline.Estimate, bestQuant *quant.Scheme) {
	in := est.In
	m := est.Memory
	fmt.Fprintf(w, "inferest fit — %s @ %s on %s · context %s\n\n",
		in.Model.Name, in.Weights.Name, in.Device.Name, group(in.Context))
	memorySection(w, est)
	fmt.Fprintf(w, "\n")
	switch {
	case !m.Known:
		fmt.Fprintf(w, "verdict: UNKNOWN — device memory capacity not specified\n")
	case m.Fits:
		fmt.Fprintf(w, "verdict: FITS — %s headroom\n", fmtBytes(m.CapacityBytes-m.TotalBytes))
	default:
		fmt.Fprintf(w, "verdict: DOES NOT FIT — %s over budget\n", fmtBytes(m.TotalBytes-m.CapacityBytes))
		if bestQuant != nil {
			fmt.Fprintf(w, "  widest quantization that fits at this context: %s (%.2f bits/weight)\n",
				bestQuant.Name, bestQuant.BitsPerWeight)
		} else {
			fmt.Fprintf(w, "  no supported quantization fits at this context\n")
		}
	}
}

// Devices writes the preset table.
func Devices(w io.Writer, devs []device.Device) {
	fmt.Fprintf(w, "%-22s %-8s %10s %12s %12s  %s\n", "device", "kind", "memory", "bandwidth", "compute", "note")
	for _, d := range devs {
		mem := "n/a"
		if d.MemoryGiB > 0 {
			mem = fmt.Sprintf("%.0f GiB", d.MemoryGiB)
		}
		fmt.Fprintf(w, "%-22s %-8s %10s %9g GB/s %6g TF16  %s\n", d.Name, d.Kind, mem, d.BandwidthGBs, d.TFLOPSFP16, d.Note)
	}
}

// Models writes the geometry preset table.
func Models(w io.Writer, models []model.Geometry) {
	fmt.Fprintf(w, "%-14s %8s %8s %7s %8s %6s %9s %8s  %s\n",
		"model", "params", "active", "layers", "d_model", "heads", "kv_heads", "context", "note")
	for _, g := range models {
		active := "="
		if g.MoE() {
			active = fmt.Sprintf("%.2fB", g.ActiveParamsB)
		}
		fmt.Fprintf(w, "%-14s %7.2fB %8s %7d %8d %6d %9d %8s  %s\n",
			g.Name, g.ParamsB, active, g.Layers, g.DModel, g.Heads, g.KVHeads, group(g.MaxContext), g.Note)
	}
}

// Quants writes the quantization tables (weights, then KV cache).
func Quants(w io.Writer, ws []quant.Scheme, kvs []quant.KVScheme) {
	fmt.Fprintf(w, "weights\n")
	fmt.Fprintf(w, "  %-6s %12s  %s\n", "name", "bits/weight", "note")
	for _, s := range ws {
		fmt.Fprintf(w, "  %-6s %12.2f  %s\n", s.Name, s.BitsPerWeight, s.Note)
	}
	fmt.Fprintf(w, "\nkv cache\n")
	fmt.Fprintf(w, "  %-6s %12s  %s\n", "name", "bytes/elem", "note")
	for _, s := range kvs {
		fmt.Fprintf(w, "  %-6s %12.2f  %s\n", s.Name, s.BytesPerElem, s.Note)
	}
}
