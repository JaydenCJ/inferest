package cli

import (
	"flag"
	"fmt"
	"io"
	"strings"

	"github.com/JaydenCJ/inferest/internal/device"
	"github.com/JaydenCJ/inferest/internal/render"
	"github.com/JaydenCJ/inferest/internal/roofline"
)

// runCompare implements `inferest compare`: one estimate per device, same
// model, quant and workload, rendered side by side.
func runCompare(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("compare", flag.ContinueOnError)
	opts := &specOpts{}
	opts.register(fs, true, "text, json or markdown")
	var deviceList string
	fs.StringVar(&deviceList, "devices", "", "comma-separated device preset names (required)")
	set, err := parseFlags(fs, args, stderr)
	if err != nil {
		return ExitUsage
	}
	if fs.NArg() > 0 {
		return usageError(stderr, fmt.Errorf("compare takes flags only, got positional %q", fs.Arg(0)))
	}
	if err := checkFormat(opts.format, "text", "json", "markdown"); err != nil {
		return usageError(stderr, err)
	}
	if strings.TrimSpace(deviceList) == "" {
		return usageError(stderr, fmt.Errorf("compare needs --devices a,b,c (see `inferest devices`)"))
	}
	if set["device"] {
		return usageError(stderr, fmt.Errorf("compare uses --devices (plural), not --device"))
	}
	// Hardware overrides are ambiguous across several devices; refuse them
	// here rather than silently applying one number to every row.
	for _, f := range []string{"memory-gb", "bandwidth", "tflops"} {
		if set[f] {
			return usageError(stderr, fmt.Errorf("--%s cannot be combined with compare; use `inferest estimate` per device instead", f))
		}
	}

	var names []string
	for _, n := range strings.Split(deviceList, ",") {
		if n = strings.TrimSpace(n); n != "" {
			names = append(names, n)
		}
	}
	if len(names) < 2 {
		return usageError(stderr, fmt.Errorf("compare needs at least two devices, got %d", len(names)))
	}

	ests := make([]roofline.Estimate, 0, len(names))
	for _, name := range names {
		if _, err := device.Lookup(name); err != nil {
			return usageError(stderr, err)
		}
		opts.deviceName = name
		in, err := opts.resolveInputs(set)
		if err != nil {
			return usageError(stderr, err)
		}
		est, err := roofline.New(in)
		if err != nil {
			return usageError(stderr, err)
		}
		ests = append(ests, est)
	}

	switch opts.format {
	case "json":
		out, err := render.CompareJSON(ests)
		if err != nil {
			return usageError(stderr, err)
		}
		fmt.Fprintln(stdout, string(out))
	case "markdown":
		render.CompareMarkdown(stdout, ests)
	default:
		render.Compare(stdout, ests)
	}
	return ExitOK
}
