# TODO

Planned work that is designed but not built yet. Each item has enough notes to
start from. Ordered roughly by value.

## crates.io release workflow

- `.github/workflows/release.yml` on tags `v*`: run the full CI matrix, check that
  the tag matches `workspace.package.version`, then `cargo publish` in dependency
  order: `qanten-sys` → `qanten` → `qanten-graphics` → `qanten-cli`
  (`qanten-mock-driver` stays unpublished). Needs a `CARGO_REGISTRY_TOKEN` secret.
- Create a GitHub release with the CHANGELOG section as notes.
- `RELEASING.md`: bump version, update CHANGELOG, tag, push.
- Before the first publish: fill in `authors`, decide on the trademark question
  (see README), and run `qanten selftest` on real hardware if possible.
