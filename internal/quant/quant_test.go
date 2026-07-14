// Tests for the quantization tables. The effective bits-per-weight figures
// are load-bearing constants — every traffic and footprint estimate is a
// multiple of them — so their invariants are pinned here.
package quant

import (
	"math"
	"strings"
	"testing"
)

func TestLookupResolvesSchemesCaseInsensitively(t *testing.T) {
	for _, in := range []string{"q4", "Q4", " q4 "} {
		s, err := Lookup(in)
		if err != nil {
			t.Fatalf("Lookup(%q): %v", in, err)
		}
		if s.Name != "q4" || s.BitsPerWeight != 4.5 {
			t.Fatalf("Lookup(%q) = %+v, want name q4 with 4.5 bpw", in, s)
		}
	}
}

func TestLookupUnknownSchemeListsKnownNames(t *testing.T) {
	_, err := Lookup("q7")
	if err == nil {
		t.Fatal("Lookup(q7) should fail")
	}
	// The error must be self-serve: it lists what would have worked.
	if !strings.Contains(err.Error(), "q4") || !strings.Contains(err.Error(), "f16") {
		t.Fatalf("error should list known schemes, got: %v", err)
	}
}

func TestLookupKVKnownAndUnknown(t *testing.T) {
	kv, err := LookupKV("f16")
	if err != nil || kv.BytesPerElem != 2.0 {
		t.Fatalf("LookupKV(f16) = %+v, %v; want 2 bytes/elem", kv, err)
	}
	if _, err := LookupKV("int3"); err == nil {
		t.Fatal("LookupKV(int3) should fail")
	}
}

func TestBlockQuantsCostMoreThanTheirNominalBits(t *testing.T) {
	// Block scales and minima are real memory traffic; a scheme that claimed
	// exactly its nominal width would flatter every estimate.
	for _, tc := range []struct {
		name    string
		nominal float64
	}{{"q2", 2}, {"q3", 3}, {"q4", 4}, {"q5", 5}, {"q6", 6}, {"q8", 8}} {
		s, err := Lookup(tc.name)
		if err != nil {
			t.Fatalf("Lookup(%s): %v", tc.name, err)
		}
		if s.BitsPerWeight <= tc.nominal {
			t.Errorf("%s: effective bpw %.2f should exceed nominal %.0f", tc.name, s.BitsPerWeight, tc.nominal)
		}
	}
}

func TestTablesOrderedWidestFirstAndNamesMatch(t *testing.T) {
	all := All()
	if len(all) < 8 {
		t.Fatalf("expected at least 8 weight schemes, got %d", len(all))
	}
	for i := 1; i < len(all); i++ {
		if all[i].BitsPerWeight > all[i-1].BitsPerWeight {
			t.Fatalf("All() not ordered widest-first: %s (%.2f) after %s (%.2f)",
				all[i].Name, all[i].BitsPerWeight, all[i-1].Name, all[i-1].BitsPerWeight)
		}
	}
	if names := Names(); len(names) != len(all) || names[0] != all[0].Name {
		t.Fatal("Names() must mirror All() order")
	}
	kvs := AllKV()
	for i := 1; i < len(kvs); i++ {
		if kvs[i].BytesPerElem > kvs[i-1].BytesPerElem {
			t.Fatalf("AllKV() not ordered widest-first at %s", kvs[i].Name)
		}
	}
	names := KVNames()
	if len(names) != len(kvs) {
		t.Fatalf("KVNames() length %d != AllKV() length %d", len(names), len(kvs))
	}
	for i := range kvs {
		if names[i] != kvs[i].Name {
			t.Fatalf("KVNames()[%d] = %s, want %s", i, names[i], kvs[i].Name)
		}
	}
}

func TestBytesForParamsExactValue(t *testing.T) {
	q4, _ := Lookup("q4")
	got := q4.BytesForParams(8.03e9)
	want := 8.03e9 * 4.5 / 8
	if math.Abs(got-want) > 1 {
		t.Fatalf("BytesForParams(8.03e9) = %g, want %g", got, want)
	}
}
