package cli

import (
	"flag"
	"fmt"
	"io"

	"github.com/JaydenCJ/inferest/internal/quant"
	"github.com/JaydenCJ/inferest/internal/render"
	"github.com/JaydenCJ/inferest/internal/roofline"
)

// runFit implements `inferest fit`: the memory verdict, with exit code 1
// when the requested configuration does not fit — usable as a shell gate.
func runFit(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("fit", flag.ContinueOnError)
	opts := &specOpts{}
	opts.register(fs, false, "text or json")
	set, err := parseFlags(fs, args, stderr)
	if err != nil {
		return ExitUsage
	}
	if fs.NArg() > 0 {
		return usageError(stderr, fmt.Errorf("fit takes flags only, got positional %q", fs.Arg(0)))
	}
	if err := checkFormat(opts.format, "text", "json"); err != nil {
		return usageError(stderr, err)
	}
	in, err := opts.resolveInputs(set)
	if err != nil {
		return usageError(stderr, err)
	}
	est, err := roofline.New(in)
	if err != nil {
		return usageError(stderr, err)
	}
	if !est.Memory.Known {
		return usageError(stderr, fmt.Errorf("fit needs a memory capacity: pick a preset with one or pass --memory-gb"))
	}

	best := widestQuantThatFits(est)

	switch opts.format {
	case "json":
		out, err := render.FitJSON(est, best)
		if err != nil {
			return usageError(stderr, err)
		}
		fmt.Fprintln(stdout, string(out))
	default:
		render.Fit(stdout, est, best)
	}
	if !est.Memory.Fits {
		return ExitNoFit
	}
	return ExitOK
}

// widestQuantThatFits re-runs the footprint math across the quant table
// (widest first) and returns the first scheme that fits at the same
// context, or nil. Only meaningful — and only computed — when the
// requested scheme does not fit.
func widestQuantThatFits(est roofline.Estimate) *quant.Scheme {
	if est.Memory.Fits {
		return nil
	}
	for _, s := range quant.All() {
		in := est.In
		in.Weights = s
		alt, err := roofline.New(in)
		if err != nil {
			continue
		}
		if alt.Memory.Fits {
			found := s
			return &found
		}
	}
	return nil
}
