#!/usr/bin/env bash
# Check the MUSA operator layer on its own, since it is not a workspace member.
#
# Two claims are verified, not just that it builds:
#   1. the catalog document and the Rust catalog list the same semantics;
#   2. no semantic resolves to a native candidate.
#
# The second is the one that matters. An empty registry is the honest state of
# this round, and a change that quietly registers something without the kernels
# to back it would make every downstream comparison meaningless.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# `cargo` lives outside PATH in a non-login shell, where the failure would
# otherwise read as `cargo: command not found` from deep inside the script.
if ! command -v cargo >/dev/null 2>&1; then
  echo "cargo is not on PATH. Add the Rust toolchain, e.g.:" >&2
  echo "  export PATH=\"\$HOME/.cargo/bin:\$PATH\"" >&2
  exit 1
fi

echo "== cargo test =="
cargo test --manifest-path "${here}/Cargo.toml"

echo
echo "== operator resolution report =="
report="$(cargo run --quiet --manifest-path "${here}/Cargo.toml" --bin operator-gaps -- --token-count 10)"
printf '%s\n' "${report}" | head -8

native="$(printf '%s' "${report}" | grep -o '"semantics_native": [0-9]*' | grep -o '[0-9]*')"
if [[ "${native}" != "0" ]]; then
  echo
  echo "FAIL: ${native} semantic(s) resolve to a native candidate, but no MUSA" >&2
  echo "kernels exist. Either register them honestly with their kernels, or fix" >&2
  echo "the registry." >&2
  exit 1
fi

echo
echo "OK: every semantic is deferred, and no value is produced for any of them."
