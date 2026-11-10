# Releasing

Releases are cut by pushing a tag. `.github/workflows/release.yml` then:

1. checks that the tag, the workspace version and the CHANGELOG agree
   (`scripts/release.py check`);
2. runs the full CI (the same `ci.yml` as every push);
3. packages and verifies every crate against the others (`cargo publish --workspace
   --dry-run` in a fresh target dir), then publishes them to crates.io in dependency
   order: `qanten-sys` → `qanten` → `qanten-graphics` → `qanten-serve` → `qanten-cli`
   (`scripts/release.py publish`; versions already on crates.io are skipped, so a failed
   run can be re-run);
4. creates a GitHub release whose notes are the CHANGELOG section.

`qanten-mock-driver`, `machine-monitor` and `qanten-bevy` are not published.

## One-time setup

- Add a crates.io API token with the "publish-new" and "publish-update" scopes as the
  repository secret `CARGO_REGISTRY_TOKEN`.
- Optionally protect the `crates-io` environment (Settings → Environments) so a person
  has to approve the publish step.

## Cutting a release

1. Bump the version in the root `Cargo.toml`, in all three places:
   `workspace.package.version` and the `version` of the `qanten-sys` and `qanten`
   entries under `workspace.dependencies`. Keep `crates/qanten-bevy/Cargo.toml` in step.
2. In `CHANGELOG.md`, rename `## X.Y.Z (unreleased)` to `## X.Y.Z (YYYY-MM-DD)`.
3. Check locally:

   ```sh
   scripts/ci.sh
   python3 scripts/release.py check vX.Y.Z
   cargo clean -p qanten-sys   # verification must not reuse an older build of the same version
   cargo publish --workspace --dry-run
   ```

4. Commit, then tag and push:

   ```sh
   git tag -a vX.Y.Z -m "qanten X.Y.Z"
   git push origin main vX.Y.Z
   ```

5. Watch the Release workflow. If publishing fails halfway, fix the cause and re-run the
   failed jobs; crates that made it to crates.io are skipped.
6. Add a new `## X.Y.(Z+1) (unreleased)` section to the CHANGELOG for the next changes.

crates.io versions cannot be deleted, only yanked (`cargo yank --version X.Y.Z <crate>`),
so a broken release is fixed with a new patch version.

## Before the first release

Not automated:

- Run `qanten selftest` and the examples on real hardware if at all possible: so far
  everything is tested on the simulator and a mock driver (see `docs/testing.md`).
