# TODO

Everything that was planned here has been built. What remains needs a decision from
the maintainers rather than code:

## Before the first crates.io release

The release workflow is ready (see [RELEASING.md](RELEASING.md)). Open points:

- Fill in `authors` in `workspace.package` (currently `qanten contributors`).
- Decide on the trademark question: the crate names use "qant", Q.ANT GmbH's name
  (the README says the project is unofficial and unaffiliated).
- Run `qanten selftest` and the examples on real Q.ANT hardware if possible.
- Add the `CARGO_REGISTRY_TOKEN` secret.
