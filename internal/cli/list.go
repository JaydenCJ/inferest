package cli

import (
	"flag"
	"fmt"
	"io"

	"github.com/JaydenCJ/inferest/internal/device"
	"github.com/JaydenCJ/inferest/internal/model"
	"github.com/JaydenCJ/inferest/internal/quant"
	"github.com/JaydenCJ/inferest/internal/render"
)

// parseListFlags handles the single --format flag the list commands share.
func parseListFlags(name string, args []string, stderr io.Writer) (string, bool) {
	fs := flag.NewFlagSet(name, flag.ContinueOnError)
	format := fs.String("format", "text", "output format (text, json)")
	fs.SetOutput(stderr)
	if err := fs.Parse(args); err != nil {
		return "", false
	}
	if fs.NArg() > 0 {
		fmt.Fprintf(stderr, "inferest: %s takes flags only, got positional %q\n", name, fs.Arg(0))
		return "", false
	}
	if err := checkFormat(*format, "text", "json"); err != nil {
		fmt.Fprintf(stderr, "inferest: %v\n", err)
		return "", false
	}
	return *format, true
}

// runDevices implements `inferest devices`.
func runDevices(args []string, stdout, stderr io.Writer) int {
	format, ok := parseListFlags("devices", args, stderr)
	if !ok {
		return ExitUsage
	}
	if format == "json" {
		out, err := render.DevicesJSON(device.All())
		if err != nil {
			return usageError(stderr, err)
		}
		fmt.Fprintln(stdout, string(out))
		return ExitOK
	}
	render.Devices(stdout, device.All())
	return ExitOK
}

// runModels implements `inferest models`.
func runModels(args []string, stdout, stderr io.Writer) int {
	format, ok := parseListFlags("models", args, stderr)
	if !ok {
		return ExitUsage
	}
	if format == "json" {
		out, err := render.ModelsJSON(model.All())
		if err != nil {
			return usageError(stderr, err)
		}
		fmt.Fprintln(stdout, string(out))
		return ExitOK
	}
	render.Models(stdout, model.All())
	return ExitOK
}

// runQuants implements `inferest quants`.
func runQuants(args []string, stdout, stderr io.Writer) int {
	format, ok := parseListFlags("quants", args, stderr)
	if !ok {
		return ExitUsage
	}
	if format == "json" {
		out, err := render.QuantsJSON(quant.All(), quant.AllKV())
		if err != nil {
			return usageError(stderr, err)
		}
		fmt.Fprintln(stdout, string(out))
		return ExitOK
	}
	render.Quants(stdout, quant.All(), quant.AllKV())
	return ExitOK
}
