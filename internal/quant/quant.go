// Package quant defines weight and KV-cache quantization schemes and their
// effective storage cost. Bits-per-weight figures are *effective*: they
// include the block scales and metadata that real block-quantization formats
// carry, which is why a "4-bit" scheme costs 4.5 bits per weight on disk and
// in memory. Traffic estimates that ignore this overhead flatter every
// device by ~10%, so inferest never does.
package quant

import (
	"fmt"
	"sort"
	"strings"
)

// Scheme is a weight quantization scheme.
type Scheme struct {
	Name          string
	BitsPerWeight float64 // effective bits per weight, incl. block metadata
	Note          string
}

// BytesForParams returns the weight footprint in bytes for a parameter count.
func (s Scheme) BytesForParams(params float64) float64 {
	return params * s.BitsPerWeight / 8
}

// schemes is the curated weight-quantization table. Names are generic on
// purpose: they describe the nominal bit width, not any runtime's private
// format zoo. Effective bits match the common block layouts (32-weight
// blocks with fp16 scales, plus per-superblock minima where applicable).
var schemes = []Scheme{
	{"f32", 32.00, "full precision; baseline, rarely deployed"},
	{"f16", 16.00, "half precision; no quantization error"},
	{"bf16", 16.00, "bfloat16; same footprint as f16"},
	{"q8", 8.50, "8-bit blocks; near-lossless"},
	{"q6", 6.56, "6-bit superblocks; negligible quality loss"},
	{"q5", 5.50, "5-bit blocks; mild quality loss"},
	{"q4", 4.50, "4-bit blocks; the common sweet spot"},
	{"q3", 3.44, "3-bit superblocks; visible quality loss"},
	{"q2", 2.63, "2-bit superblocks; heavy quality loss"},
}

// KVScheme is a KV-cache element precision.
type KVScheme struct {
	Name         string
	BytesPerElem float64
	Note         string
}

// kvSchemes is the KV-cache precision table.
var kvSchemes = []KVScheme{
	{"f32", 4.0, "full-precision cache; baseline"},
	{"f16", 2.0, "the default almost everywhere"},
	{"q8", 1.0, "8-bit cache; usually indistinguishable"},
	{"q4", 0.5, "4-bit cache; long-context quality risk"},
}

// Lookup resolves a weight scheme by name (case-insensitive).
func Lookup(name string) (Scheme, error) {
	key := strings.ToLower(strings.TrimSpace(name))
	for _, s := range schemes {
		if s.Name == key {
			return s, nil
		}
	}
	return Scheme{}, fmt.Errorf("unknown quantization %q (known: %s)", name, strings.Join(Names(), ", "))
}

// LookupKV resolves a KV-cache scheme by name (case-insensitive).
func LookupKV(name string) (KVScheme, error) {
	key := strings.ToLower(strings.TrimSpace(name))
	for _, s := range kvSchemes {
		if s.Name == key {
			return s, nil
		}
	}
	return KVScheme{}, fmt.Errorf("unknown kv-cache precision %q (known: %s)", name, strings.Join(KVNames(), ", "))
}

// All returns the weight schemes ordered from widest to narrowest, name as
// tie-break, so every listing is deterministic.
func All() []Scheme {
	out := make([]Scheme, len(schemes))
	copy(out, schemes)
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].BitsPerWeight != out[j].BitsPerWeight {
			return out[i].BitsPerWeight > out[j].BitsPerWeight
		}
		return out[i].Name < out[j].Name
	})
	return out
}

// AllKV returns the KV-cache schemes ordered from widest to narrowest.
func AllKV() []KVScheme {
	out := make([]KVScheme, len(kvSchemes))
	copy(out, kvSchemes)
	sort.SliceStable(out, func(i, j int) bool {
		return out[i].BytesPerElem > out[j].BytesPerElem
	})
	return out
}

// Names lists the weight scheme names in All() order.
func Names() []string {
	all := All()
	names := make([]string, len(all))
	for i, s := range all {
		names[i] = s.Name
	}
	return names
}

// KVNames lists the KV scheme names in AllKV() order.
func KVNames() []string {
	all := AllKV()
	names := make([]string, len(all))
	for i, s := range all {
		names[i] = s.Name
	}
	return names
}
