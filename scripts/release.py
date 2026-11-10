#!/usr/bin/env python3
"""Release helper used by .github/workflows/release.yml (standard library only).

    release.py check vX.Y.Z    tag, workspace version, inter-crate versions and
                               CHANGELOG heading agree; prints the version
    release.py notes X.Y.Z     the CHANGELOG section for X.Y.Z (release notes)
    release.py publish         cargo publish every crate in dependency order,
                               skipping versions already on crates.io (so a
                               failed run can simply be re-run)

See RELEASING.md for the whole procedure.
"""

import pathlib
import re
import subprocess
import sys
import tomllib
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Dependency order. qanten-mock-driver, machine-monitor and qanten-bevy are
# not published (publish = false).
CRATES = ["qanten-sys", "qanten", "qanten-graphics", "qanten-serve", "qanten-cli"]


def fail(msg: str) -> None:
    print(f"release.py: {msg}", file=sys.stderr)
    sys.exit(1)


def workspace() -> dict:
    return tomllib.loads((ROOT / "Cargo.toml").read_text())


def version() -> str:
    return workspace()["workspace"]["package"]["version"]


def changelog_section(ver: str) -> tuple[str, str]:
    """(heading, body) of the CHANGELOG section for `ver`."""
    text = (ROOT / "CHANGELOG.md").read_text()
    m = re.search(rf"^## {re.escape(ver)}\b(.*)$", text, re.M)
    if not m:
        fail(f"CHANGELOG.md has no '## {ver}' section")
    start = m.end()
    nxt = re.search(r"^## ", text[start:], re.M)
    body = text[start : start + nxt.start()] if nxt else text[start:]
    return m.group(0), body.strip()


def check(tag: str) -> None:
    ver = version()
    if tag.removeprefix("v") != ver:
        fail(f"tag {tag} does not match workspace.package.version {ver}")
    deps = workspace()["workspace"]["dependencies"]
    for name in ("qanten-sys", "qanten"):
        if deps[name].get("version") != ver:
            fail(f"workspace.dependencies.{name}.version is {deps[name].get('version')}, expected {ver}")
    heading, body = changelog_section(ver)
    if "unreleased" in heading.lower():
        fail(f"CHANGELOG heading '{heading}' still says unreleased; give it the release date")
    if not re.search(r"\(\d{4}-\d{2}-\d{2}\)", heading):
        fail(f"CHANGELOG heading '{heading}' needs a date: '## {ver} (YYYY-MM-DD)'")
    if not body:
        fail(f"CHANGELOG section for {ver} is empty")
    for crate in CRATES:
        manifest = tomllib.loads((ROOT / "crates" / crate / "Cargo.toml").read_text())
        if manifest["package"].get("publish", True) is False:
            fail(f"{crate} has publish = false")
    print(ver)


def notes(ver: str) -> None:
    print(changelog_section(ver)[1])


def published(crate: str, ver: str) -> bool:
    req = urllib.request.Request(
        f"https://crates.io/api/v1/crates/{crate}/{ver}",
        headers={"User-Agent": "qanten release script (https://github.com/aprilthepink/qanten)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30):
            return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise


def publish() -> None:
    ver = version()
    for crate in CRATES:
        if published(crate, ver):
            print(f"{crate} {ver} is already on crates.io, skipping")
            continue
        print(f"publishing {crate} {ver}", flush=True)
        # cargo waits until the crate is in the index before returning, so the
        # next crate can depend on it.
        subprocess.run(["cargo", "publish", "--locked", "-p", crate], cwd=ROOT, check=True)


def main() -> None:
    args = sys.argv[1:]
    match args:
        case ["check", tag]:
            check(tag)
        case ["notes", ver]:
            notes(ver)
        case ["publish"]:
            publish()
        case _:
            print(__doc__)
            sys.exit(2)


if __name__ == "__main__":
    main()
