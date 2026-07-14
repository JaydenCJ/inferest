package render

import (
	"fmt"
	"io"

	"github.com/JaydenCJ/inferest/internal/roofline"
)

// EstimateMarkdown renders the estimate as PR-pasteable Markdown tables.
func EstimateMarkdown(w io.Writer, est roofline.Estimate) {
	in := est.In
	fmt.Fprintf(w, "### inferest estimate — %s @ %s on %s\n\n", in.Model.Name, in.Weights.Name, in.Device.Name)
	fmt.Fprintf(w, "- device: %s\n", deviceLine(in.Device))
	fmt.Fprintf(w, "- model: %s\n", modelLine(in.Model))
	fmt.Fprintf(w, "- quant: %s\n\n", quantLine(in.Weights, in.KVCache, est.Memory.KVBytesPerToken))

	m := est.Memory
	fmt.Fprintf(w, "| Memory @ context %s | |\n|---|---|\n", group(in.Context))
	fmt.Fprintf(w, "| Weights | %s |\n", fmtBytes(m.WeightBytes))
	fmt.Fprintf(w, "| KV cache | %s |\n", fmtBytes(m.KVBytes))
	fmt.Fprintf(w, "| Overhead | %s |\n", fmtBytes(m.OverheadBytes))
	if m.Known {
		verdict := "fits"
		if !m.Fits {
			verdict = "**does not fit**"
		}
		fmt.Fprintf(w, "| Total | %s / %s — %s (%s) |\n", fmtBytes(m.TotalBytes), fmtBytes(m.CapacityBytes), verdict, fmtPct(m.UsedFraction))
		fmt.Fprintf(w, "| Max context | ≈ %s tokens |\n", group(m.MaxContext))
	} else {
		fmt.Fprintf(w, "| Total | %s (capacity unknown) |\n", fmtBytes(m.TotalBytes))
	}

	fmt.Fprintf(w, "\n| Decode t/s | Conservative | Expected | Optimistic |\n|---|---|---|---|\n")
	for _, p := range est.Decode.Points {
		label := "empty context"
		if p.ContextTokens > 0 {
			label = "@ " + group(p.ContextTokens) + " tokens"
		}
		fmt.Fprintf(w, "| %s | %s | %s | %s |\n", label,
			fmtTPS(p.TPS.Conservative), fmtTPS(p.TPS.Expected), fmtTPS(p.TPS.Optimistic))
	}

	pf := est.Prefill
	fmt.Fprintf(w, "\n| Prefill (%s-token prompt) | Conservative | Expected | Optimistic |\n|---|---|---|---|\n", group(pf.PromptTokens))
	fmt.Fprintf(w, "| Speed (t/s) | %s | %s | %s |\n",
		fmtTPS(pf.TPS.Conservative), fmtTPS(pf.TPS.Expected), fmtTPS(pf.TPS.Optimistic))
	fmt.Fprintf(w, "| Time to first token | %s | %s | %s |\n",
		fmtSeconds(pf.TTFTSeconds.Conservative), fmtSeconds(pf.TTFTSeconds.Expected), fmtSeconds(pf.TTFTSeconds.Optimistic))
}

// CompareMarkdown renders the device comparison as one Markdown table.
func CompareMarkdown(w io.Writer, ests []roofline.Estimate) {
	if len(ests) == 0 {
		return
	}
	in := ests[0].In
	fmt.Fprintf(w, "### inferest compare — %s @ %s · context %s · prompt %s\n\n",
		in.Model.Name, in.Weights.Name, group(in.Context), group(in.Prompt))
	fmt.Fprintf(w, "| Device | Fit | Decode t/s (expected) | Range (c–o) | Prefill t/s | TTFT |\n")
	fmt.Fprintf(w, "|---|---|---|---|---|---|\n")
	for _, est := range ests {
		m := est.Memory
		fit := "?"
		if m.Known {
			fit = fmtPct(m.UsedFraction)
			if !m.Fits {
				fit = "**does not fit**"
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
		fmt.Fprintf(w, "| %s | %s | %s | %s | %s | %s |\n",
			est.In.Device.Name, fit, decodeCol, rangeCol, prefillCol, ttftCol)
	}
}
