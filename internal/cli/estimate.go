package cli

import (
	"flag"
	"fmt"
	"io"

	"github.com/JaydenCJ/inferest/internal/render"
	"github.com/JaydenCJ/inferest/internal/roofline"
)

// runEstimate implements `inferest estimate`.
func runEstimate(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("estimate", flag.ContinueOnError)
	opts := &specOpts{}
	opts.register(fs, true, "text, json or markdown")
	set, err := parseFlags(fs, args, stderr)
	if err != nil {
		return ExitUsage
	}
	if fs.NArg() > 0 {
		return usageError(stderr, fmt.Errorf("estimate takes flags only, got positional %q", fs.Arg(0)))
	}
	if err := checkFormat(opts.format, "text", "json", "markdown"); err != nil {
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
	switch opts.format {
	case "json":
		out, err := render.EstimateJSON(est)
		if err != nil {
			return usageError(stderr, err)
		}
		fmt.Fprintln(stdout, string(out))
	case "markdown":
		render.EstimateMarkdown(stdout, est)
	default:
		render.Estimate(stdout, est)
	}
	return ExitOK
}
