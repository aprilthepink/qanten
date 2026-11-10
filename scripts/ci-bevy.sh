#!/usr/bin/env bash
# The `bevy` job of .github/workflows/ci.yml: checks crates/qanten-bevy, a
# separate workspace because Bevy is a large build. The windowed check needs
# libasound2-dev libudev-dev libwayland-dev libxkbcommon-dev on Linux.
set -euo pipefail
cd "$(dirname "$0")/../crates/qanten-bevy"
export RUSTFLAGS="-D warnings"
cargo fmt --check
cargo clippy -q --all-targets --features gltf
cargo test -q --features gltf
cargo check -q --features windowed --example ocean
echo "ci-bevy.sh: all checks passed"
