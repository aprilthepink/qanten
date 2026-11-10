# qanten-bevy

[Bevy](https://bevy.org) 0.17 plugins for [`qanten-graphics`](../qanten-graphics): NPU-evaluated
Gerstner oceans with buoyancy, skinned colliders for picking and collision, and
light-probe baking into textures. Runs on a Q.ANT photonic NPU, or on qanten's
built-in simulator without one.

```sh
cargo test --manifest-path crates/qanten-bevy/Cargo.toml
cargo run --release --manifest-path crates/qanten-bevy/Cargo.toml --features windowed --example ocean
```

Guide: [`docs/graphics/bevy.md`](../../docs/graphics/bevy.md).

This is a separate Cargo workspace (Bevy is a large build) and is not published yet.
