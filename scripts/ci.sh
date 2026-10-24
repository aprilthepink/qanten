#!/usr/bin/env bash
# Runs the same checks as .github/workflows/ci.yml, stopping at the first failure.
set -euo pipefail
cd "$(dirname "$0")/.."
export RUSTFLAGS="-D warnings"
cargo fmt --all --check
cargo clippy -q --workspace --all-targets --all-features
cargo clippy -q -p qanten --no-default-features --all-targets
cargo test -q --workspace --all-features
cargo test -q -p qanten --no-default-features
cargo build -q -p qanten-mock-driver
QANTEN_DRIVER_LIB="$PWD/target/debug/libqant_native_computing_driver.so" cargo test -q -p qanten --test driver_ffi
RUSTDOCFLAGS="-D warnings" cargo doc -q --workspace --all-features --no-deps
if rustup toolchain list | grep -q '^1.88'; then cargo +1.88 check -q --workspace --all-features --all-targets; fi
python3 scripts/check_docs.py > /dev/null
echo "ci.sh: all checks passed"
