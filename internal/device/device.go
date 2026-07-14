// Package device describes inference hardware by the three numbers that
// actually bound single-stream LLM speed: memory capacity, memory
// bandwidth, and dense half-precision compute. Presets carry public
// spec-sheet figures; all of them can be overridden per run, because the
// point of a closed-form estimator is arguing about the inputs, not
// trusting ours.
package device

import (
	"fmt"
	"sort"
	"strings"
)

// Kinds of hardware inferest knows how to describe.
const (
	KindGPU     = "gpu"     // discrete GPU with its own VRAM
	KindUnified = "unified" // SoC with unified memory (CPU+GPU share it)
	KindCPU     = "cpu"     // CPU inference from system RAM
	KindSBC     = "sbc"     // single-board computer
)

// Device is an inference target reduced to its roofline-relevant numbers.
//
// Unit conventions (deliberate, documented in docs/method.md):
//   - MemoryGiB is binary gibibytes, because VRAM is marketed that way.
//   - BandwidthGBs is decimal GB/s, because spec sheets quote it that way.
//   - TFLOPSFP16 is dense (non-sparse) half-precision teraFLOPS.
type Device struct {
	Name         string
	Kind         string
	MemoryGiB    float64 // 0 = unknown / user-configured
	BandwidthGBs float64
	TFLOPSFP16   float64
	Note         string
}

// MemoryBytes converts capacity to bytes; 0 means unknown.
func (d Device) MemoryBytes() float64 { return d.MemoryGiB * (1 << 30) }

// BandwidthBytesPerSec converts spec-sheet GB/s to bytes per second.
func (d Device) BandwidthBytesPerSec() float64 { return d.BandwidthGBs * 1e9 }

// FLOPS converts TFLOPS to FLOP/s.
func (d Device) FLOPS() float64 { return d.TFLOPSFP16 * 1e12 }

// Validate rejects devices that cannot bound anything.
func (d Device) Validate() error {
	switch {
	case d.BandwidthGBs <= 0:
		return fmt.Errorf("device %q: bandwidth must be > 0", d.Name)
	case d.TFLOPSFP16 <= 0:
		return fmt.Errorf("device %q: compute (TFLOPS) must be > 0", d.Name)
	case d.MemoryGiB < 0:
		return fmt.Errorf("device %q: memory cannot be negative", d.Name)
	}
	return nil
}

// presets carry public spec-sheet numbers. Compute is dense FP16 with FP32
// accumulate (the convention inference kernels actually run at); memory for
// configurable machines is a common configuration and is meant to be
// overridden with --memory-gb.
var presets = []Device{
	{Name: "h100-sxm", Kind: KindGPU, MemoryGiB: 80, BandwidthGBs: 3350, TFLOPSFP16: 989, Note: "data-center reference point"},
	{Name: "a100-80gb", Kind: KindGPU, MemoryGiB: 80, BandwidthGBs: 2039, TFLOPSFP16: 312, Note: "previous-gen data-center"},
	{Name: "rtx-5090", Kind: KindGPU, MemoryGiB: 32, BandwidthGBs: 1792, TFLOPSFP16: 209.5, Note: "GDDR7 flagship"},
	{Name: "rtx-4090", Kind: KindGPU, MemoryGiB: 24, BandwidthGBs: 1008, TFLOPSFP16: 165.2, Note: "the enthusiast baseline"},
	{Name: "rtx-4080", Kind: KindGPU, MemoryGiB: 16, BandwidthGBs: 716.8, TFLOPSFP16: 97.5, Note: ""},
	{Name: "rtx-4070", Kind: KindGPU, MemoryGiB: 12, BandwidthGBs: 504.2, TFLOPSFP16: 58.3, Note: ""},
	{Name: "rtx-3090", Kind: KindGPU, MemoryGiB: 24, BandwidthGBs: 936, TFLOPSFP16: 71.2, Note: "used-market VRAM favourite"},
	{Name: "rtx-3060-12gb", Kind: KindGPU, MemoryGiB: 12, BandwidthGBs: 360, TFLOPSFP16: 25.5, Note: "budget entry point"},
	{Name: "rx-7900-xtx", Kind: KindGPU, MemoryGiB: 24, BandwidthGBs: 960, TFLOPSFP16: 122.8, Note: "RDNA3 flagship"},
	{Name: "apple-m2", Kind: KindUnified, MemoryGiB: 16, BandwidthGBs: 100, TFLOPSFP16: 7.2, Note: "memory as configured; override --memory-gb"},
	{Name: "apple-m3-pro", Kind: KindUnified, MemoryGiB: 18, BandwidthGBs: 150, TFLOPSFP16: 14.2, Note: "memory as configured; override --memory-gb"},
	{Name: "apple-m3-max", Kind: KindUnified, MemoryGiB: 48, BandwidthGBs: 400, TFLOPSFP16: 28.4, Note: "memory as configured; override --memory-gb"},
	{Name: "apple-m4", Kind: KindUnified, MemoryGiB: 16, BandwidthGBs: 120, TFLOPSFP16: 8.5, Note: "memory as configured; override --memory-gb"},
	{Name: "apple-m4-pro", Kind: KindUnified, MemoryGiB: 24, BandwidthGBs: 273, TFLOPSFP16: 18.4, Note: "memory as configured; override --memory-gb"},
	{Name: "apple-m4-max", Kind: KindUnified, MemoryGiB: 48, BandwidthGBs: 546, TFLOPSFP16: 36.8, Note: "memory as configured; override --memory-gb"},
	{Name: "jetson-orin-nano-8gb", Kind: KindSBC, MemoryGiB: 8, BandwidthGBs: 68, TFLOPSFP16: 10, Note: "shared CPU/GPU LPDDR5"},
	{Name: "raspberry-pi-5", Kind: KindSBC, MemoryGiB: 8, BandwidthGBs: 17.1, TFLOPSFP16: 0.25, Note: "quad Cortex-A76, LPDDR4X-4267"},
	{Name: "ddr5-desktop", Kind: KindCPU, MemoryGiB: 64, BandwidthGBs: 89.6, TFLOPSFP16: 1.0, Note: "dual-channel DDR5-5600; override to match your build"},
	{Name: "ddr4-desktop", Kind: KindCPU, MemoryGiB: 32, BandwidthGBs: 51.2, TFLOPSFP16: 0.6, Note: "dual-channel DDR4-3200; override to match your build"},
}

// Lookup resolves a preset by name (case-insensitive).
func Lookup(name string) (Device, error) {
	key := strings.ToLower(strings.TrimSpace(name))
	for _, d := range presets {
		if d.Name == key {
			return d, nil
		}
	}
	return Device{}, fmt.Errorf("unknown device %q (see `inferest devices` for the %d built-ins, or describe yours with --bandwidth/--tflops/--memory-gb)", name, len(presets))
}

// All returns the presets ordered by bandwidth descending, name as
// tie-break — fastest decode first, the order every listing uses.
func All() []Device {
	out := make([]Device, len(presets))
	copy(out, presets)
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].BandwidthGBs != out[j].BandwidthGBs {
			return out[i].BandwidthGBs > out[j].BandwidthGBs
		}
		return out[i].Name < out[j].Name
	})
	return out
}

// Names lists preset names in All() order.
func Names() []string {
	all := All()
	names := make([]string, len(all))
	for i, d := range all {
		names[i] = d.Name
	}
	return names
}
