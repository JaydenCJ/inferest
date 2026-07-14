// In-process integration tests for the CLI: Run(argv, stdout, stderr) is
// exercised exactly as main() would, asserting on real output and exit
// codes without building a binary. No filesystem, no network, no clock.
package cli

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/JaydenCJ/inferest/internal/version"
)

// run invokes the CLI and returns (exit, stdout, stderr).
func run(args ...string) (int, string, string) {
	var out, errBuf bytes.Buffer
	code := Run(args, &out, &errBuf)
	return code, out.String(), errBuf.String()
}

func TestVersionPrintsTheSemverUnderEveryAlias(t *testing.T) {
	code, out, _ := run("version")
	if code != ExitOK {
		t.Fatalf("exit = %d, want 0", code)
	}
	if want := "inferest " + version.Version + "\n"; out != want {
		t.Fatalf("output = %q, want %q", out, want)
	}
	for _, alias := range []string{"--version", "-v"} {
		code, out, _ := run(alias)
		if code != ExitOK || !strings.Contains(out, version.Version) {
			t.Errorf("%s: exit %d, out %q", alias, code, out)
		}
	}
}

func TestHelpExitsZeroAndListsCommands(t *testing.T) {
	code, out, _ := run("help")
	if code != ExitOK {
		t.Fatalf("exit = %d, want 0", code)
	}
	for _, cmd := range []string{"estimate", "compare", "fit", "devices", "models", "quants"} {
		if !strings.Contains(out, cmd) {
			t.Errorf("help missing command %q", cmd)
		}
	}
}

func TestNoArgsIsAUsageError(t *testing.T) {
	code, _, errOut := run()
	if code != ExitUsage {
		t.Fatalf("exit = %d, want %d", code, ExitUsage)
	}
	if !strings.Contains(errOut, "usage:") {
		t.Fatal("bare invocation should print usage to stderr")
	}
}

func TestUnknownCommandExitsUsage(t *testing.T) {
	code, _, errOut := run("benchmark")
	if code != ExitUsage || !strings.Contains(errOut, "unknown command") {
		t.Fatalf("exit = %d, stderr = %q", code, errOut)
	}
}

func TestEstimateHappyPathText(t *testing.T) {
	code, out, errOut := run("estimate", "--device", "rtx-4090", "--model", "8b", "--quant", "q4")
	if code != ExitOK {
		t.Fatalf("exit = %d, stderr = %q", code, errOut)
	}
	for _, want := range []string{"inferest estimate — 8b @ q4 on rtx-4090", "FITS", "decode speed", "prefill"} {
		if !strings.Contains(out, want) {
			t.Errorf("output missing %q\n---\n%s", want, out)
		}
	}
}

func TestEstimateJSONParsesWithEnvelope(t *testing.T) {
	code, out, _ := run("estimate", "--device", "rtx-4090", "--model", "8b", "--format", "json")
	if code != ExitOK {
		t.Fatalf("exit = %d", code)
	}
	var doc struct {
		Tool          string `json:"tool"`
		SchemaVersion int    `json:"schema_version"`
		Memory        struct {
			Fits bool `json:"fits"`
		} `json:"memory"`
	}
	if err := json.Unmarshal([]byte(out), &doc); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if doc.Tool != "inferest" || doc.SchemaVersion != 1 || !doc.Memory.Fits {
		t.Fatalf("bad envelope/verdict: %+v", doc)
	}
}

func TestEstimateMarkdownFormat(t *testing.T) {
	code, out, _ := run("estimate", "--device", "rtx-4090", "--model", "8b", "--format", "markdown")
	if code != ExitOK || !strings.Contains(out, "### inferest estimate") {
		t.Fatalf("exit = %d, out = %q", code, out)
	}
}

func TestEstimateRejectsUnknownNames(t *testing.T) {
	// Every name-shaped flag must fail closed with a self-serve error.
	cases := []struct {
		name    string
		args    []string
		wantErr string
	}{
		{"device", []string{"estimate", "--device", "gtx-9090", "--model", "8b"}, "unknown device"},
		{"model", []string{"estimate", "--device", "rtx-4090", "--model", "9000b"}, "unknown model"},
		{"quant", []string{"estimate", "--device", "rtx-4090", "--model", "8b", "--quant", "q7"}, "unknown quantization"},
		{"kv-quant", []string{"estimate", "--device", "rtx-4090", "--model", "8b", "--kv-quant", "int3"}, "unknown kv-cache precision"},
		{"format", []string{"estimate", "--device", "rtx-4090", "--model", "8b", "--format", "yaml"}, `unknown --format "yaml"`},
	}
	for _, tc := range cases {
		code, _, errOut := run(tc.args...)
		if code != ExitUsage || !strings.Contains(errOut, tc.wantErr) {
			t.Errorf("%s: exit = %d, stderr = %q (want %q)", tc.name, code, errOut, tc.wantErr)
		}
	}
}

func TestEstimateRejectsPositionalArguments(t *testing.T) {
	code, _, errOut := run("estimate", "rtx-4090")
	if code != ExitUsage || !strings.Contains(errOut, "positional") {
		t.Fatalf("exit = %d, stderr = %q", code, errOut)
	}
}

func TestCustomDeviceNeedsBandwidthAndTflops(t *testing.T) {
	code, _, errOut := run("estimate", "--model", "8b", "--bandwidth", "100")
	if code != ExitUsage || !strings.Contains(errOut, "--tflops") {
		t.Fatalf("exit = %d, stderr = %q", code, errOut)
	}
}

func TestCustomDeviceEndToEnd(t *testing.T) {
	// A hypothetical 200 GB/s, 20 TFLOPS, 16 GiB board — the "sanity-check
	// hardware you do not own" use case.
	code, out, errOut := run("estimate", "--model", "8b", "--quant", "q4",
		"--bandwidth", "200", "--tflops", "20", "--memory-gb", "16")
	if code != ExitOK {
		t.Fatalf("exit = %d, stderr = %q", code, errOut)
	}
	if !strings.Contains(out, "on custom") || !strings.Contains(out, "FITS") {
		t.Fatalf("custom-device output unexpected:\n%s", out)
	}
}

func TestCustomModelListsEveryMissingFlag(t *testing.T) {
	code, _, errOut := run("estimate", "--device", "rtx-4090", "--params", "8")
	if code != ExitUsage {
		t.Fatalf("exit = %d", code)
	}
	for _, flag := range []string{"--layers", "--d-model", "--heads", "--kv-heads", "--ffn", "--vocab"} {
		if !strings.Contains(errOut, flag) {
			t.Errorf("error should name missing %s, got: %q", flag, errOut)
		}
	}
}

func TestCustomModelEndToEnd(t *testing.T) {
	code, out, errOut := run("estimate", "--device", "rtx-4090", "--quant", "q4",
		"--params", "8.03", "--layers", "32", "--d-model", "4096",
		"--heads", "32", "--kv-heads", "8", "--ffn", "14336", "--vocab", "128256")
	if code != ExitOK {
		t.Fatalf("exit = %d, stderr = %q", code, errOut)
	}
	if !strings.Contains(out, "custom · 8.03B params") {
		t.Fatalf("custom-model output unexpected:\n%s", out)
	}
}

func TestPresetOverrideChangesTheEstimate(t *testing.T) {
	// Overriding a preset's bandwidth must move the decode number; this is
	// how users model overclocks and cut-down variants.
	parse := func(out string) float64 {
		var doc struct {
			Decode struct {
				Points []struct {
					TPS struct {
						Expected float64 `json:"expected"`
					} `json:"tps"`
				} `json:"points"`
			} `json:"decode"`
		}
		if err := json.Unmarshal([]byte(out), &doc); err != nil {
			t.Fatalf("invalid JSON: %v", err)
		}
		return doc.Decode.Points[0].TPS.Expected
	}
	_, base, _ := run("estimate", "--device", "rtx-4090", "--model", "8b", "--format", "json")
	_, halved, _ := run("estimate", "--device", "rtx-4090", "--model", "8b", "--format", "json", "--bandwidth", "504")
	b, h := parse(base), parse(halved)
	if h >= b || h < b*0.49 || h > b*0.51 {
		t.Fatalf("halving bandwidth should halve bandwidth-bound decode: %g → %g", b, h)
	}
}

func TestPromptDefaultsShrinkToSmallContexts(t *testing.T) {
	// --context 512 without --prompt must not trip the prompt<=context rule.
	code, out, errOut := run("estimate", "--device", "rtx-4090", "--model", "8b", "--context", "512")
	if code != ExitOK {
		t.Fatalf("exit = %d, stderr = %q", code, errOut)
	}
	if !strings.Contains(out, "prefill (512-token prompt)") {
		t.Fatalf("prompt should default to min(1024, context):\n%s", out)
	}
}

func TestEstimateRejectsImpossibleWorkloads(t *testing.T) {
	cases := []struct {
		name    string
		extra   []string
		wantErr string
	}{
		{"prompt beyond context", []string{"--context", "2048", "--prompt", "4096"}, "cannot exceed context"},
		{"zero context", []string{"--context", "0"}, "--context"},
		{"efficiency above 1", []string{"--bw-eff", "1.5"}, "efficiency"},
	}
	for _, tc := range cases {
		args := append([]string{"estimate", "--device", "rtx-4090", "--model", "8b"}, tc.extra...)
		code, _, errOut := run(args...)
		if code != ExitUsage || !strings.Contains(errOut, tc.wantErr) {
			t.Errorf("%s: exit = %d, stderr = %q (want %q)", tc.name, code, errOut, tc.wantErr)
		}
	}
}

func TestCompareKeepsDeviceOrder(t *testing.T) {
	code, out, errOut := run("compare", "--devices", "raspberry-pi-5,rtx-4090", "--model", "1b")
	if code != ExitOK {
		t.Fatalf("exit = %d, stderr = %q", code, errOut)
	}
	pi := strings.Index(out, "raspberry-pi-5")
	gpu := strings.Index(out, "rtx-4090")
	if pi == -1 || gpu == -1 || pi > gpu {
		t.Fatalf("rows must keep the caller's order:\n%s", out)
	}
}

func TestCompareRejectsBadDeviceSets(t *testing.T) {
	cases := []struct {
		name    string
		args    []string
		wantErr string
	}{
		{"single device", []string{"compare", "--devices", "rtx-4090", "--model", "8b"}, "at least two"},
		{"no devices flag", []string{"compare", "--model", "8b"}, "--devices"},
		{"singular --device", []string{"compare", "--devices", "rtx-4090,apple-m4", "--device", "rtx-4090", "--model", "8b"}, "plural"},
		// One --memory-gb silently applied to every row would fabricate data.
		{"hardware override", []string{"compare", "--devices", "rtx-4090,apple-m4", "--model", "8b", "--memory-gb", "48"}, "--memory-gb"},
		{"unknown device", []string{"compare", "--devices", "rtx-4090,gtx-9090", "--model", "8b"}, "gtx-9090"},
	}
	for _, tc := range cases {
		code, _, errOut := run(tc.args...)
		if code != ExitUsage || !strings.Contains(errOut, tc.wantErr) {
			t.Errorf("%s: exit = %d, stderr = %q (want %q)", tc.name, code, errOut, tc.wantErr)
		}
	}
}

func TestCompareJSONHasOneResultPerDevice(t *testing.T) {
	code, out, _ := run("compare", "--devices", "rtx-4090,apple-m4,raspberry-pi-5", "--model", "1b", "--format", "json")
	if code != ExitOK {
		t.Fatalf("exit = %d", code)
	}
	var doc struct {
		Results []json.RawMessage `json:"results"`
	}
	if err := json.Unmarshal([]byte(out), &doc); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if len(doc.Results) != 3 {
		t.Fatalf("results = %d, want 3", len(doc.Results))
	}
}

func TestFitExitsZeroWhenItFits(t *testing.T) {
	code, out, errOut := run("fit", "--device", "rtx-4090", "--model", "8b", "--quant", "q4")
	if code != ExitOK {
		t.Fatalf("exit = %d, stderr = %q", code, errOut)
	}
	if !strings.Contains(out, "verdict: FITS") {
		t.Fatalf("fit output:\n%s", out)
	}
}

func TestFitExitsOneWhenItDoesNotFit(t *testing.T) {
	code, out, _ := run("fit", "--device", "rtx-4090", "--model", "70b", "--quant", "q4")
	if code != ExitNoFit {
		t.Fatalf("exit = %d, want %d (the shell-gate contract)", code, ExitNoFit)
	}
	if !strings.Contains(out, "DOES NOT FIT") {
		t.Fatalf("fit output:\n%s", out)
	}
}

func TestFitSuggestsTheWidestQuantThatFitsInBothFormats(t *testing.T) {
	// 70b @ f16 needs ≈131 GiB and misses an 80 GiB card; q8 (≈70 GiB of
	// weights) is the widest scheme that fits at this context.
	code, out, _ := run("fit", "--device", "a100-80gb", "--model", "70b", "--quant", "f16")
	if code != ExitNoFit {
		t.Fatalf("exit = %d, want %d", code, ExitNoFit)
	}
	if !strings.Contains(out, "widest quantization that fits at this context: q8") {
		t.Fatalf("fit should suggest q8:\n%s", out)
	}

	code, out, _ = run("fit", "--device", "a100-80gb", "--model", "70b", "--quant", "f16", "--format", "json")
	if code != ExitNoFit {
		t.Fatalf("json exit = %d", code)
	}
	var doc struct {
		Widest string `json:"widest_quant_that_fits"`
		Memory struct {
			Fits bool `json:"fits"`
		} `json:"memory"`
	}
	if err := json.Unmarshal([]byte(out), &doc); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if doc.Memory.Fits || doc.Widest != "q8" {
		t.Fatalf("fit JSON: fits=%v widest=%q, want false/q8", doc.Memory.Fits, doc.Widest)
	}
}

func TestFitWithoutCapacityExitsUsage(t *testing.T) {
	code, _, errOut := run("fit", "--model", "8b", "--bandwidth", "100", "--tflops", "10")
	if code != ExitUsage || !strings.Contains(errOut, "--memory-gb") {
		t.Fatalf("exit = %d, stderr = %q", code, errOut)
	}
}

func TestDevicesListsPresetsInBandwidthOrder(t *testing.T) {
	code, out, _ := run("devices")
	if code != ExitOK {
		t.Fatalf("exit = %d", code)
	}
	h100 := strings.Index(out, "h100-sxm")
	pi := strings.Index(out, "raspberry-pi-5")
	if h100 == -1 || pi == -1 || h100 > pi {
		t.Fatalf("devices should list fastest first:\n%s", out)
	}
}

func TestDevicesJSONParses(t *testing.T) {
	code, out, _ := run("devices", "--format", "json")
	if code != ExitOK {
		t.Fatalf("exit = %d", code)
	}
	var doc struct {
		Devices []struct {
			Name         string  `json:"name"`
			BandwidthGBs float64 `json:"bandwidth_gb_s"`
		} `json:"devices"`
	}
	if err := json.Unmarshal([]byte(out), &doc); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if len(doc.Devices) < 15 || doc.Devices[0].BandwidthGBs == 0 {
		t.Fatalf("devices JSON incomplete: %d rows", len(doc.Devices))
	}
}

func TestModelsListsPresetsSmallestFirst(t *testing.T) {
	code, out, _ := run("models")
	if code != ExitOK {
		t.Fatalf("exit = %d", code)
	}
	small := strings.Index(out, "1b")
	big := strings.Index(out, "70b")
	if small == -1 || big == -1 || small > big {
		t.Fatalf("models should list smallest first:\n%s", out)
	}
}

func TestQuantsListShowsEffectiveBits(t *testing.T) {
	code, out, _ := run("quants")
	if code != ExitOK || !strings.Contains(out, "4.50") {
		t.Fatalf("exit = %d, out:\n%s", code, out)
	}
}

func TestListCommandsRejectBadInput(t *testing.T) {
	for _, cmd := range []string{"devices", "models", "quants"} {
		code, _, errOut := run(cmd, "--format", "csv")
		if code != ExitUsage || !strings.Contains(errOut, "unknown --format") {
			t.Errorf("%s: exit = %d, stderr = %q", cmd, code, errOut)
		}
	}
	code, _, errOut := run("devices", "extra")
	if code != ExitUsage || !strings.Contains(errOut, "positional") {
		t.Fatalf("positional: exit = %d, stderr = %q", code, errOut)
	}
}

func TestMoEPresetEndToEnd(t *testing.T) {
	// MoE must show total-vs-active split in the header and much faster
	// decode than a dense model of the same footprint would get.
	code, out, errOut := run("estimate", "--device", "apple-m4-max", "--model", "moe-8x7b", "--quant", "q4")
	if code != ExitOK {
		t.Fatalf("exit = %d, stderr = %q", code, errOut)
	}
	if !strings.Contains(out, "46.70B total / 12.88B active") {
		t.Fatalf("MoE header missing total/active split:\n%s", out)
	}
}

func TestKVQuantFlagChangesTheCacheLine(t *testing.T) {
	_, f16, _ := run("estimate", "--device", "rtx-4090", "--model", "8b")
	_, q8, _ := run("estimate", "--device", "rtx-4090", "--model", "8b", "--kv-quant", "q8")
	if !strings.Contains(f16, "128.0 KiB per context token") {
		t.Fatalf("f16 cache line missing:\n%s", f16)
	}
	if !strings.Contains(q8, "64.0 KiB per context token") {
		t.Fatalf("q8 cache should halve the per-token line:\n%s", q8)
	}
}
