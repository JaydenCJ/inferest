// Command inferest estimates tokens-per-second bounds for LLM inference
// from device bandwidth, compute and model geometry — no hardware, no
// model download, no benchmark run required.
package main

import (
	"os"

	"github.com/JaydenCJ/inferest/internal/cli"
)

func main() {
	os.Exit(cli.Run(os.Args[1:], os.Stdout, os.Stderr))
}
