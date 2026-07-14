// Tests for the renderers. Formatting helpers are pinned exactly; the
// text/JSON/Markdown surfaces are checked for structure and for agreeing
// with each other, since all three must be views of the same estimate.
package render

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/JaydenCJ/inferest/internal/device"
	"github.com/JaydenCJ/inferest/internal/model"
	"github.com/JaydenCJ/inferest/internal/quant"
	"github.com/JaydenCJ/inferest/internal/roofline"
)

// estimateFor builds a real estimate from presets.
func estimateFor(t *testing.T, dev, mdl, q string, ctx int) roofline.Estimate {
	t.Helper()
	d, err := device.Lookup(dev)
	if err != nil {
		t.Fatalf("device.Lookup: %v", err)
	}
	g, err := model.Lookup(mdl)
	if err != nil {
		t.Fatalf("model.Lookup: %v", err)
	}
	w, err := quant.Lookup(q)
	if err != nil {
		t.Fatalf("quant.Lookup: %v", err)
	}
	kv, err := quant.LookupKV("f16")
	if err != nil {
		t.Fatalf("quant.LookupKV: %v", err)
	}
	prompt := 1024
	if ctx < prompt {
		prompt = ctx
	}
	est, err := roofline.New(roofline.Inputs{Device: d, Model: g, Weights: w, KVCache: kv, Context: ctx, Prompt: prompt})
	if err != nil {
		t.Fatalf("roofline.New: %v", err)
	}
	return est
}

func TestByteAndCountFormatting(t *testing.T) {
	byteCases := []struct {
		in   float64
		want string
	}{
		{512, "512 B"},
		{131072, "128.0 KiB"},
		{5 * 1 << 20, "5.0 MiB"},
		{4.5168750e9, "4.21 GiB"},
	}
	for _, tc := range byteCases {
		if got := fmtBytes(tc.in); got != tc.want {
			t.Errorf("fmtBytes(%g) = %q, want %q", tc.in, got, tc.want)
		}
	}
	groupCases := []struct {
		in   int
		want string
	}{{0, "0"}, {999, "999"}, {8192, "8,192"}, {131072, "131,072"}, {-8192, "-8,192"}}
	for _, tc := range groupCases {
		if got := group(tc.in); got != tc.want {
			t.Errorf("group(%d) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestRateAndTimeFormatting(t *testing.T) {
	// SBC decode rates need decimals; prefill rates in the thousands don't.
	tpsCases := []struct {
		in   float64
		want string
	}{{2.136, "2.14"}, {34.25, "34.2"}, {4047.9, "4048"}}
	for _, tc := range tpsCases {
		if got := fmtTPS(tc.in); got != tc.want {
			t.Errorf("fmtTPS(%g) = %q, want %q", tc.in, got, tc.want)
		}
	}
	secCases := []struct {
		in   float64
		want string
	}{{0.253, "253 ms"}, {2.27, "2.27 s"}, {167.2, "167.2 s"}}
	for _, tc := range secCases {
		if got := fmtSeconds(tc.in); got != tc.want {
			t.Errorf("fmtSeconds(%g) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestEstimateTextContainsEverySection(t *testing.T) {
	var buf bytes.Buffer
	Estimate(&buf, estimateFor(t, "rtx-4090", "8b", "q4", 8192))
	out := buf.String()
	for _, want := range []string{
		"inferest estimate", "device ", "model ", "quant ",
		"memory @ context 8,192", "decode speed", "prefill", "max context",
		"bound: memory bandwidth", "bound: compute",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("estimate text missing %q\n---\n%s", want, out)
		}
	}
}

func TestEstimateTextShowsFitVerdicts(t *testing.T) {
	var fits, noFit bytes.Buffer
	Estimate(&fits, estimateFor(t, "rtx-4090", "8b", "q4", 8192))
	Estimate(&noFit, estimateFor(t, "rtx-4090", "70b", "q4", 8192))
	if !strings.Contains(fits.String(), "FITS") || strings.Contains(fits.String(), "DOES NOT FIT") {
		t.Error("8b on 24 GiB should render FITS")
	}
	if !strings.Contains(noFit.String(), "DOES NOT FIT") {
		t.Error("70b on 24 GiB should render DOES NOT FIT")
	}
	if !strings.Contains(noFit.String(), "weights alone exceed capacity") {
		t.Error("no-fit report should explain that weights alone exceed capacity")
	}
}

func TestEstimateJSONIsValidAndCarriesTheEnvelope(t *testing.T) {
	out, err := EstimateJSON(estimateFor(t, "rtx-4090", "8b", "q4", 8192))
	if err != nil {
		t.Fatalf("EstimateJSON: %v", err)
	}
	var doc map[string]any
	if err := json.Unmarshal(out, &doc); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if doc["tool"] != "inferest" || doc["schema_version"] != float64(SchemaVersion) || doc["command"] != "estimate" {
		t.Fatalf("bad envelope: tool=%v schema=%v command=%v", doc["tool"], doc["schema_version"], doc["command"])
	}
	for _, key := range []string{"inputs", "memory", "decode", "prefill"} {
		if _, ok := doc[key]; !ok {
			t.Errorf("estimate JSON missing top-level key %q", key)
		}
	}
	if strings.Contains(string(out), "NaN") {
		t.Fatal("JSON must never contain NaN")
	}
}

func TestEstimateJSONAgreesWithTextNumbers(t *testing.T) {
	// The two surfaces must be views of the same struct: pick one number
	// (KV bytes per token for the 8b class = 131072) and find it in both.
	est := estimateFor(t, "rtx-4090", "8b", "q4", 8192)
	out, err := EstimateJSON(est)
	if err != nil {
		t.Fatalf("EstimateJSON: %v", err)
	}
	if !strings.Contains(string(out), `"kv_bytes_per_token": 131072`) {
		t.Fatal("JSON should carry kv_bytes_per_token 131072 for the 8b class")
	}
	var buf bytes.Buffer
	Estimate(&buf, est)
	if !strings.Contains(buf.String(), "128.0 KiB per context token") {
		t.Fatal("text should render the same 131072 B as 128.0 KiB")
	}
}

func TestCompareTextDashesOutNonFittingDevices(t *testing.T) {
	ests := []roofline.Estimate{
		estimateFor(t, "rtx-4090", "70b", "q4", 8192),
		estimateFor(t, "apple-m4-max", "70b", "q4", 8192),
	}
	var buf bytes.Buffer
	Compare(&buf, ests)
	out := buf.String()
	if !strings.Contains(out, "DOES NOT FIT") {
		t.Fatalf("4090 row should say DOES NOT FIT:\n%s", out)
	}
	for _, line := range strings.Split(out, "\n") {
		if strings.HasPrefix(line, "rtx-4090") && !strings.Contains(line, "—") {
			t.Fatalf("non-fitting row should dash out speed columns: %q", line)
		}
	}
}

func TestCompareJSONKeepsRowOrder(t *testing.T) {
	ests := []roofline.Estimate{
		estimateFor(t, "raspberry-pi-5", "1b", "q4", 4096),
		estimateFor(t, "rtx-4090", "1b", "q4", 4096),
	}
	out, err := CompareJSON(ests)
	if err != nil {
		t.Fatalf("CompareJSON: %v", err)
	}
	var doc struct {
		Results []struct {
			Device struct {
				Name string `json:"name"`
			} `json:"device"`
		} `json:"results"`
	}
	if err := json.Unmarshal(out, &doc); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if len(doc.Results) != 2 || doc.Results[0].Device.Name != "raspberry-pi-5" || doc.Results[1].Device.Name != "rtx-4090" {
		t.Fatalf("rows must keep caller order, got %+v", doc.Results)
	}
}

func TestFitTextNamesTheWidestQuantThatFits(t *testing.T) {
	est := estimateFor(t, "rtx-4090", "32b", "f16", 8192) // 32b f16 ≈ 61 GiB: no fit
	q5, err := quant.Lookup("q5")
	if err != nil {
		t.Fatalf("quant.Lookup: %v", err)
	}
	var buf bytes.Buffer
	Fit(&buf, est, &q5)
	out := buf.String()
	if !strings.Contains(out, "DOES NOT FIT") || !strings.Contains(out, "widest quantization that fits at this context: q5") {
		t.Fatalf("fit text should carry the q5 suggestion:\n%s", out)
	}
}

func TestFitTextHeadroomWhenItFits(t *testing.T) {
	var buf bytes.Buffer
	Fit(&buf, estimateFor(t, "rtx-4090", "8b", "q4", 8192), nil)
	if !strings.Contains(buf.String(), "verdict: FITS") || !strings.Contains(buf.String(), "headroom") {
		t.Fatalf("fitting verdict should report headroom:\n%s", buf.String())
	}
}

func TestMarkdownSurfacesEmitTables(t *testing.T) {
	var buf bytes.Buffer
	EstimateMarkdown(&buf, estimateFor(t, "rtx-4090", "8b", "q4", 8192))
	out := buf.String()
	if !strings.Contains(out, "### inferest estimate") {
		t.Fatal("markdown should start with an H3 header")
	}
	if !strings.Contains(out, "| Decode t/s | Conservative | Expected | Optimistic |") {
		t.Fatalf("markdown missing the decode table header:\n%s", out)
	}
	if !strings.Contains(out, "|---|---|---|---|") {
		t.Fatal("markdown missing table separators")
	}

	ests := []roofline.Estimate{
		estimateFor(t, "rtx-4090", "8b", "q4", 8192),
		estimateFor(t, "apple-m4", "8b", "q4", 8192),
	}
	var cmp bytes.Buffer
	CompareMarkdown(&cmp, ests)
	if !strings.Contains(cmp.String(), "| rtx-4090 |") || !strings.Contains(cmp.String(), "| apple-m4 |") {
		t.Fatalf("markdown compare should have a row per device:\n%s", cmp.String())
	}
}

func TestListRenderersShowEveryRow(t *testing.T) {
	var buf bytes.Buffer
	Devices(&buf, device.All())
	for _, d := range device.All() {
		if !strings.Contains(buf.String(), d.Name) {
			t.Errorf("devices listing missing %s", d.Name)
		}
	}
	var qbuf bytes.Buffer
	Quants(&qbuf, quant.All(), quant.AllKV())
	out := qbuf.String()
	if !strings.Contains(out, "weights") || !strings.Contains(out, "kv cache") {
		t.Fatalf("quants listing should show both tables:\n%s", out)
	}
	if !strings.Contains(out, "4.50") {
		t.Fatal("quants listing should show effective bits (q4 → 4.50)")
	}
}
