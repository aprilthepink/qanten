#!/usr/bin/env python3
"""Compile (and run) every ```rust snippet in docs/ against the workspace crates.

Mark special snippets with an HTML comment on the line before the fence:
  <!-- doccheck: ignore -->        skip
  <!-- doccheck: no_run -->        compile only
  <!-- doccheck: compile_fail -->  must fail to compile
Snippets without `fn main` are wrapped in one returning Result.
"""
import json, pathlib, re, subprocess, sys, tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
FENCE = re.compile(r'(?:<!-- doccheck: (\w+) -->\n)?```rust\n(.*?)```', re.S)

def main() -> int:
    work = pathlib.Path(tempfile.mkdtemp(prefix="qanten-doccheck-"))
    (work / "src").mkdir()
    (work / "examples").mkdir()
    (work / "src/main.rs").write_text("fn main() {}\n")
    (work / "Cargo.toml").write_text(f"""[package]
name = "doccheck"
version = "0.0.0"
edition = "2024"
publish = false

[dependencies]
qanten = {{ path = "{ROOT}/crates/qanten", features = ["full"] }}
qanten-sys = {{ path = "{ROOT}/crates/qanten-sys" }}
qanten-graphics = {{ path = "{ROOT}/crates/qanten-graphics" }}
tracing = "0.1"

[workspace]
""")
    snippets = []
    for md in sorted((ROOT / "docs").rglob("*.md")):
        for m in FENCE.finditer(md.read_text()):
            mode, code = m.group(1) or "run", m.group(2)
            if mode == "ignore":
                continue
            if "fn main" not in code:
                code = "fn main() -> Result<(), Box<dyn std::error::Error>> {\n" + code + "\nOk(())\n}\n"
            name = f"s{len(snippets):03}"
            (work / "examples" / f"{name}.rs").write_text(code)
            snippets.append((name, md.relative_to(ROOT), mode))

    env_target = ["--target-dir", str(ROOT / "target" / "doccheck")]
    failures = 0
    for name, src, mode in snippets:
        build = subprocess.run(["cargo", "build", "-q", "--example", name, *env_target],
                               cwd=work, capture_output=True, text=True)
        ok = build.returncode != 0 if mode == "compile_fail" else build.returncode == 0
        detail = build.stderr
        if ok and mode == "run":
            run = subprocess.run([str(ROOT / "target/doccheck/debug/examples" / name)],
                                 capture_output=True, text=True, timeout=300)
            ok, detail = run.returncode == 0, run.stderr
        if not ok or "--quiet" not in sys.argv:
            print(f"{'ok  ' if ok else 'FAIL'} {src} [{mode}]")
        if not ok:
            failures += 1
            print(detail[-2000:])
    print(f"{len(snippets) - failures}/{len(snippets)} snippets ok")
    return 1 if failures else 0

if __name__ == "__main__":
    sys.exit(main())
