"""The package version, from rust/Cargo.toml (pyproject.toml reads it from there).

  python .github/version.py            print it; exit 1 if rust/Cargo.lock disagrees
  python .github/version.py --notes    print its CHANGELOG.md section; exit 1 if there is none
"""

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: str) -> dict:
    return tomllib.loads((ROOT / path).read_text())


version = load("rust/Cargo.toml")["package"]["version"]
lock = next(p["version"] for p in load("rust/Cargo.lock")["package"] if p["name"] == "symbfitrs")
if lock != version:
    sys.exit(f"rust/Cargo.lock has {lock}, rust/Cargo.toml {version}: run cargo update --workspace")
if "--notes" in sys.argv:
    changelog = (ROOT / "CHANGELOG.md").read_text()
    m = re.search(rf"^## {re.escape(version)} \(.*?\)\n(.*?)(?=^## |\Z)", changelog, re.M | re.S)
    if not m:
        sys.exit(f"CHANGELOG.md has no '## {version} (YYYY-MM-DD)' section")
    print(m.group(1).strip())
else:
    print(version)
