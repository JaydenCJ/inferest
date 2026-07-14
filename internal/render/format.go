// Package render turns roofline estimates into text, JSON and Markdown.
// Renderers never compute — they only format what internal/roofline already
// derived, so every number is identical across the three formats.
package render

import (
	"fmt"
	"strconv"
)

// fmtBytes renders a byte count at a human scale (binary units).
func fmtBytes(b float64) string {
	switch {
	case b >= 1<<30:
		return fmt.Sprintf("%.2f GiB", b/(1<<30))
	case b >= 1<<20:
		return fmt.Sprintf("%.1f MiB", b/(1<<20))
	case b >= 1<<10:
		return fmt.Sprintf("%.1f KiB", b/(1<<10))
	default:
		return fmt.Sprintf("%.0f B", b)
	}
}

// fmtTPS renders a tokens-per-second figure with scale-appropriate
// precision: SBC decode rates deserve two decimals, prefill rates none.
func fmtTPS(t float64) string {
	switch {
	case t >= 1000:
		return fmt.Sprintf("%.0f", t)
	case t >= 10:
		return fmt.Sprintf("%.1f", t)
	default:
		return fmt.Sprintf("%.2f", t)
	}
}

// fmtSeconds renders a duration, switching to milliseconds below 1 s.
func fmtSeconds(s float64) string {
	switch {
	case s >= 10:
		return fmt.Sprintf("%.1f s", s)
	case s >= 1:
		return fmt.Sprintf("%.2f s", s)
	default:
		return fmt.Sprintf("%.0f ms", s*1000)
	}
}

// fmtPct renders a fraction as a percentage.
func fmtPct(f float64) string { return fmt.Sprintf("%.1f%%", f*100) }

// group renders an integer with thousands separators: 163840 → "163,840".
func group(n int) string {
	s := strconv.Itoa(n)
	neg := false
	if len(s) > 0 && s[0] == '-' {
		neg, s = true, s[1:]
	}
	out := make([]byte, 0, len(s)+len(s)/3)
	for i, c := range []byte(s) {
		if i > 0 && (len(s)-i)%3 == 0 {
			out = append(out, ',')
		}
		out = append(out, c)
	}
	if neg {
		return "-" + string(out)
	}
	return string(out)
}
