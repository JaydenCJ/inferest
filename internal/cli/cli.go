// Package cli implements the inferest command-line interface. Run takes
// argv and two writers and returns an exit code, so the entire surface is
// testable in-process without building a binary.
package cli

import (
	"flag"
	"fmt"
	"io"
	"strings"

	"github.com/JaydenCJ/inferest/internal/device"
	"github.com/JaydenCJ/inferest/internal/model"
	"github.com/JaydenCJ/inferest/internal/quant"
	"github.com/JaydenCJ/inferest/internal/roofline"
	"github.com/JaydenCJ/inferest/internal/version"
)

// Exit codes. Documented in the README; `fit` uses ExitNoFit as its
// machine-readable verdict so shell scripts can gate on it.
const (
	ExitOK    = 0
	ExitNoFit = 1
	ExitUsage = 2
)

// Run dispatches argv and returns the process exit code.
func Run(args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		usage(stderr)
		return ExitUsage
	}
	switch args[0] {
	case "estimate":
		return runEstimate(args[1:], stdout, stderr)
	case "compare":
		return runCompare(args[1:], stdout, stderr)
	case "fit":
		return runFit(args[1:], stdout, stderr)
	case "devices":
		return runDevices(args[1:], stdout, stderr)
	case "models":
		return runModels(args[1:], stdout, stderr)
	case "quants":
		return runQuants(args[1:], stdout, stderr)
	case "version", "--version", "-v":
		fmt.Fprintf(stdout, "inferest %s\n", version.Version)
		return ExitOK
	case "help", "--help", "-h":
		usage(stdout)
		return ExitOK
	default:
		fmt.Fprintf(stderr, "inferest: unknown command %q\n\n", args[0])
		usage(stderr)
		return ExitUsage
	}
}

func usage(w io.Writer) {
	fmt.Fprint(w, `usage: inferest <command> [flags]

commands:
  estimate   full speed and memory report for one device + model + quant
  compare    the same estimate across several devices, as a table
  fit        memory verdict: breakdown, max context, widest quant that fits
  devices    list built-in device presets
  models     list built-in model geometry presets
  quants     list quantization schemes and their effective storage cost
  version    print the version

Run 'inferest <command> -h' for the flags of each command.
Exit codes: 0 ok · 1 fit verdict is "does not fit" · 2 usage error
`)
}

// specOpts collects the flags shared by estimate, compare and fit: which
// device, which model geometry, which quantization, and the workload shape.
type specOpts struct {
	// device selection / overrides
	deviceName string
	memoryGB   float64
	bandwidth  float64
	tflops     float64
	// model selection / overrides
	modelName     string
	params        float64
	activeParams  float64
	layers        int
	dModel        int
	heads         int
	kvHeads       int
	headDim       int
	ffn           int
	vocab         int
	experts       int
	activeExperts int
	tied          bool
	// quantization and workload
	quantName string
	kvQuant   string
	context   int
	prompt    int
	bwEff     float64
	mfu       float64
	format    string
}

// register wires the shared flags into a FlagSet. formats documents the
// output formats the command accepts (validated later by checkFormat).
// Usage strings avoid backquotes on purpose: the flag package would treat
// the quoted words as the value placeholder in -h output.
func (o *specOpts) register(fs *flag.FlagSet, withPrompt bool, formats string) {
	fs.StringVar(&o.deviceName, "device", "", "device preset name; list with 'inferest devices'")
	fs.Float64Var(&o.memoryGB, "memory-gb", 0, "device memory in GiB (override preset / describe custom)")
	fs.Float64Var(&o.bandwidth, "bandwidth", 0, "memory bandwidth in GB/s (override preset / describe custom)")
	fs.Float64Var(&o.tflops, "tflops", 0, "dense fp16 compute in TFLOPS (override preset / describe custom)")
	fs.StringVar(&o.modelName, "model", "", "model preset name; list with 'inferest models'")
	fs.Float64Var(&o.params, "params", 0, "total parameters in billions (custom model)")
	fs.Float64Var(&o.activeParams, "active-params", 0, "active parameters in billions (MoE; default = params)")
	fs.IntVar(&o.layers, "layers", 0, "transformer layers (custom model)")
	fs.IntVar(&o.dModel, "d-model", 0, "hidden width d_model (custom model)")
	fs.IntVar(&o.heads, "heads", 0, "query heads (custom model)")
	fs.IntVar(&o.kvHeads, "kv-heads", 0, "key/value heads (custom model)")
	fs.IntVar(&o.headDim, "head-dim", 0, "per-head dimension (default d_model/heads)")
	fs.IntVar(&o.ffn, "ffn", 0, "feed-forward width, per expert for MoE (custom model)")
	fs.IntVar(&o.vocab, "vocab", 0, "vocabulary size (custom model)")
	fs.IntVar(&o.experts, "experts", 0, "expert count (custom MoE model)")
	fs.IntVar(&o.activeExperts, "active-experts", 0, "experts routed per token (custom MoE model)")
	fs.BoolVar(&o.tied, "tied", false, "input embedding shared with the output head")
	fs.StringVar(&o.quantName, "quant", "q4", "weight quantization scheme; list with 'inferest quants'")
	fs.StringVar(&o.kvQuant, "kv-quant", "f16", "kv-cache precision; list with 'inferest quants'")
	fs.IntVar(&o.context, "context", 8192, "context window to plan for, in tokens")
	if withPrompt {
		fs.IntVar(&o.prompt, "prompt", 0, "prompt length for prefill/TTFT (default min(1024, context))")
	}
	fs.Float64Var(&o.bwEff, "bw-eff", 0, "override bandwidth efficiency with a single value in (0,1]")
	fs.Float64Var(&o.mfu, "mfu", 0, "override compute utilization with a single value in (0,1]")
	fs.StringVar(&o.format, "format", "text", "output format ("+formats+")")
}

// parseFlags parses args and records which flags were explicitly set, so
// presets can be selectively overridden.
func parseFlags(fs *flag.FlagSet, args []string, stderr io.Writer) (map[string]bool, error) {
	fs.SetOutput(stderr)
	if err := fs.Parse(args); err != nil {
		return nil, err
	}
	set := map[string]bool{}
	fs.Visit(func(f *flag.Flag) { set[f.Name] = true })
	return set, nil
}

// resolveDevice builds the device from a preset plus overrides, or from
// scratch when no preset is named.
func (o *specOpts) resolveDevice(set map[string]bool) (device.Device, error) {
	var dev device.Device
	if o.deviceName != "" {
		found, err := device.Lookup(o.deviceName)
		if err != nil {
			return device.Device{}, err
		}
		dev = found
	} else {
		if !set["bandwidth"] || !set["tflops"] {
			return device.Device{}, fmt.Errorf("no --device given: a custom device needs at least --bandwidth and --tflops (--memory-gb enables the fit verdict)")
		}
		dev = device.Device{Name: "custom", Kind: "custom"}
	}
	if set["memory-gb"] {
		dev.MemoryGiB = o.memoryGB
	}
	if set["bandwidth"] {
		dev.BandwidthGBs = o.bandwidth
	}
	if set["tflops"] {
		dev.TFLOPSFP16 = o.tflops
	}
	return dev, nil
}

// resolveModel builds the geometry from a preset plus overrides, or from
// scratch when no preset is named.
func (o *specOpts) resolveModel(set map[string]bool) (model.Geometry, error) {
	var g model.Geometry
	if o.modelName != "" {
		found, err := model.Lookup(o.modelName)
		if err != nil {
			return model.Geometry{}, err
		}
		g = found
	} else {
		var missing []string
		for _, req := range []struct {
			name string
			ok   bool
		}{
			{"params", set["params"]}, {"layers", set["layers"]}, {"d-model", set["d-model"]},
			{"heads", set["heads"]}, {"kv-heads", set["kv-heads"]}, {"ffn", set["ffn"]}, {"vocab", set["vocab"]},
		} {
			if !req.ok {
				missing = append(missing, "--"+req.name)
			}
		}
		if len(missing) > 0 {
			return model.Geometry{}, fmt.Errorf("no --model given: a custom model also needs %s", strings.Join(missing, ", "))
		}
		g = model.Geometry{Name: "custom"}
	}
	if set["params"] {
		g.ParamsB = o.params
	}
	if set["active-params"] {
		g.ActiveParamsB = o.activeParams
	}
	if set["layers"] {
		g.Layers = o.layers
	}
	if set["d-model"] {
		g.DModel = o.dModel
	}
	if set["heads"] {
		g.Heads = o.heads
	}
	if set["kv-heads"] {
		g.KVHeads = o.kvHeads
	}
	if set["head-dim"] {
		g.HeadDim = o.headDim
	}
	if set["ffn"] {
		g.FFN = o.ffn
	}
	if set["vocab"] {
		g.Vocab = o.vocab
	}
	if set["experts"] {
		g.Experts = o.experts
	}
	if set["active-experts"] {
		g.ActiveExperts = o.activeExperts
	}
	if set["tied"] {
		g.Tied = o.tied
	}
	return g, nil
}

// resolveInputs assembles a full roofline request from the parsed flags.
func (o *specOpts) resolveInputs(set map[string]bool) (roofline.Inputs, error) {
	dev, err := o.resolveDevice(set)
	if err != nil {
		return roofline.Inputs{}, err
	}
	g, err := o.resolveModel(set)
	if err != nil {
		return roofline.Inputs{}, err
	}
	w, err := quant.Lookup(o.quantName)
	if err != nil {
		return roofline.Inputs{}, err
	}
	kv, err := quant.LookupKV(o.kvQuant)
	if err != nil {
		return roofline.Inputs{}, err
	}
	if o.context < 1 {
		return roofline.Inputs{}, fmt.Errorf("--context must be >= 1, got %d", o.context)
	}
	prompt := o.prompt
	if prompt == 0 {
		prompt = 1024
		if o.context < prompt {
			prompt = o.context
		}
	}
	return roofline.Inputs{
		Device: dev, Model: g, Weights: w, KVCache: kv,
		Context: o.context, Prompt: prompt,
		BandwidthEff: o.bwEff, MFU: o.mfu,
	}, nil
}

// checkFormat validates --format against what a command supports.
func checkFormat(format string, allowed ...string) error {
	for _, a := range allowed {
		if format == a {
			return nil
		}
	}
	return fmt.Errorf("unknown --format %q (supported: %s)", format, strings.Join(allowed, ", "))
}

// usageError prints err and returns ExitUsage; the single funnel keeps
// error formatting identical across commands.
func usageError(stderr io.Writer, err error) int {
	fmt.Fprintf(stderr, "inferest: %v\n", err)
	return ExitUsage
}
