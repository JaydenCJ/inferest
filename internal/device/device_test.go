// Tests for the device catalogue: unit conversions (the GiB-vs-GB/s split
// is deliberate and easy to regress), preset hygiene, and lookup ergonomics.
package device

import (
	"strings"
	"testing"
)

func TestLookupResolvesDevicesCaseInsensitively(t *testing.T) {
	for _, in := range []string{"rtx-4090", "RTX-4090", " rtx-4090 "} {
		d, err := Lookup(in)
		if err != nil {
			t.Fatalf("Lookup(%q): %v", in, err)
		}
		if d.MemoryGiB != 24 || d.BandwidthGBs != 1008 {
			t.Fatalf("rtx-4090 = %+v, want 24 GiB / 1008 GB/s", d)
		}
	}
}

func TestLookupUnknownDevicePointsAtSelfHelp(t *testing.T) {
	_, err := Lookup("gtx-9090")
	if err == nil {
		t.Fatal("Lookup(gtx-9090) should fail")
	}
	// The error must tell the user both escape hatches: the preset listing
	// and the custom-device flags.
	if !strings.Contains(err.Error(), "inferest devices") || !strings.Contains(err.Error(), "--bandwidth") {
		t.Fatalf("error should mention `inferest devices` and --bandwidth, got: %v", err)
	}
}

func TestAllSortedByBandwidthDescending(t *testing.T) {
	all := All()
	if len(all) < 15 {
		t.Fatalf("expected at least 15 presets, got %d", len(all))
	}
	for i := 1; i < len(all); i++ {
		if all[i].BandwidthGBs > all[i-1].BandwidthGBs {
			t.Fatalf("All() not sorted by bandwidth: %s after %s", all[i].Name, all[i-1].Name)
		}
	}
	names := Names()
	if len(names) != len(all) {
		t.Fatalf("Names() length %d != All() length %d", len(names), len(all))
	}
	for i := range all {
		if names[i] != all[i].Name {
			t.Fatalf("Names()[%d] = %s, want %s", i, names[i], all[i].Name)
		}
	}
}

func TestEveryPresetValidatesAndIsLowercase(t *testing.T) {
	for _, d := range All() {
		if err := d.Validate(); err != nil {
			t.Errorf("preset %s failed validation: %v", d.Name, err)
		}
		if d.Name != strings.ToLower(d.Name) {
			t.Errorf("preset %q must be lowercase for case-insensitive lookup", d.Name)
		}
	}
}

func TestNoDuplicatePresetNames(t *testing.T) {
	seen := map[string]bool{}
	for _, d := range All() {
		if seen[d.Name] {
			t.Errorf("duplicate preset name %q", d.Name)
		}
		seen[d.Name] = true
	}
}

func TestUnitConversionsUseTheDocumentedConventions(t *testing.T) {
	// Memory is binary (GiB), bandwidth and compute are decimal — mixing
	// them up shifts every estimate by ~7%, so pin both.
	d := Device{Name: "conv", MemoryGiB: 24, BandwidthGBs: 1000, TFLOPSFP16: 100}
	if got, want := d.MemoryBytes(), 24.0*1024*1024*1024; got != want {
		t.Fatalf("MemoryBytes() = %g, want %g (binary)", got, want)
	}
	if got, want := d.BandwidthBytesPerSec(), 1e12; got != want {
		t.Fatalf("BandwidthBytesPerSec() = %g, want %g (decimal)", got, want)
	}
	if got, want := d.FLOPS(), 1e14; got != want {
		t.Fatalf("FLOPS() = %g, want %g (decimal)", got, want)
	}
}

func TestValidateRejectsUnusableDevices(t *testing.T) {
	cases := []struct {
		name string
		d    Device
	}{
		{"zero bandwidth", Device{Name: "x", BandwidthGBs: 0, TFLOPSFP16: 1}},
		{"zero compute", Device{Name: "x", BandwidthGBs: 100, TFLOPSFP16: 0}},
		{"negative memory", Device{Name: "x", BandwidthGBs: 100, TFLOPSFP16: 1, MemoryGiB: -1}},
	}
	for _, tc := range cases {
		if err := tc.d.Validate(); err == nil {
			t.Errorf("%s: Validate() should fail", tc.name)
		}
	}
	// But capacity 0 must remain a legal "unknown": SoCs and DIY builds
	// have user-chosen memory, and speed estimates still work without a
	// fit verdict.
	unknown := Device{Name: "x", BandwidthGBs: 100, TFLOPSFP16: 1, MemoryGiB: 0}
	if err := unknown.Validate(); err != nil {
		t.Fatalf("zero memory should validate as unknown, got %v", err)
	}
	if unknown.MemoryBytes() != 0 {
		t.Fatal("unknown memory should convert to 0 bytes")
	}
}
