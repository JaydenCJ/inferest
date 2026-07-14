// Package version pins the single source of truth for the release version.
package version

// Version is the semantic version of inferest. It must match CHANGELOG.md
// and the module header in go.mod; scripts/smoke.sh asserts on it.
const Version = "0.1.0"
