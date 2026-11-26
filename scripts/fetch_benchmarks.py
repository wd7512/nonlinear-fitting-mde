"""Download external benchmark data into data/external/ (git-ignored).

  nist             27 NIST StRD .dat files from itl.nist.gov (public domain)
  fitbenchmarking  FitBenchmarking problem files (BSD-3), pinned commit
  sagefit          SAGE-Fit code & Structure-Progress Bank data, pinned commit
  stringer2019     Stringer et al. (2019) V1 drifting-gratings recording (figshare, CC BY-NC 4.0,
                   860 MB, SHA-256 pinned) -> data/external/stringer2019/gratings.npz

Usage: uv run python scripts/fetch_benchmarks.py [nist] [fitbenchmarking] [sagefit] [stringer2019]
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "external"
REPOS = {
    "fitbenchmarking": (
        "https://github.com/fitbenchmarking/fitbenchmarking.git",
        "c894cdafc67581c11877f45ebb62182dd833889e",
    ),
    "sagefit": (
        "https://github.com/CAS-CLab/SAGE-Fit.git",
        "870237c423e9310a5e90b37677f49e464423cbf4",
    ),
}


def fetch_nist() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from symbolic_regression_rust.benchmarks.nist import NAMES, URL

    dest = OUT / "nist"
    dest.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        target = dest / f"{name}.dat"
        if target.exists():
            continue
        req = urllib.request.Request(
            URL.format(name=name), headers={"User-Agent": "symbolic-regression-rust"}
        )
        with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310 - fixed https URL
            target.write_bytes(r.read())
        print(f"  {target.relative_to(ROOT)}")


def fetch_repo(key: str) -> None:
    url, commit = REPOS[key]
    dest = OUT / key
    if not dest.exists():
        subprocess.run(
            ["git", "clone", "--quiet", "--filter=blob:none", url, str(dest)], check=True
        )
    subprocess.run(["git", "-C", str(dest), "checkout", "--quiet", commit], check=True)
    print(f"  {dest.relative_to(ROOT)} @ {commit[:10]}")


STRINGER_URL = "https://ndownloader.figshare.com/files/15479246"
STRINGER_FILE = "gratings_drifting_GT1_2019_04_12_1.npy"
STRINGER_SHA256 = "dbfbb9cd7764bd15feddf8c0d79a8e17a51d956a978c01f87e54b472b42e1b57"
ACTIVE_SHARE, CONCENTRATION = 0.4, 0.55  # cell eligibility, as in the EDGAR study


def fetch_stringer2019(dest: Path = OUT / "stringer2019") -> None:
    """Orientation-tuned cells: active (> 0) on more than 40% of trials with orientation
    concentration > 0.55; each keeps its first 40%-of-trials active trials; responses scaled to
    norm 100 per cell. The recording is a pickled dict, so its checksum is verified first."""
    dest.mkdir(parents=True, exist_ok=True)
    rec = dest / STRINGER_FILE
    if not rec.exists():
        print(f"  downloading {STRINGER_URL} (860 MB)")
        urllib.request.urlretrieve(STRINGER_URL, rec)
    sha = hashlib.sha256()
    with rec.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            sha.update(block)
    if sha.hexdigest() != STRINGER_SHA256:
        raise SystemExit(f"{rec}: checksum differs from the pinned figshare file")
    d = np.load(rec, allow_pickle=True).item()
    resp, angles = np.asarray(d["sresp"]), np.asarray(d["istim"])
    active = resp > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        conc = np.abs((np.exp(2j * angles)[None, :] * resp).sum(1) / resp.sum(1))
    cells = np.flatnonzero((active.mean(1) > ACTIVE_SHARE) & (conc > CONCENTRATION))
    n = int(resp.shape[1] * ACTIVE_SHARE)
    trials = np.stack([np.flatnonzero(active[c])[:n] for c in cells])
    y = resp[cells[:, None], trials].astype(np.float64)
    y = 100 * y / np.linalg.norm(y, axis=1, keepdims=True)
    out = dest / "gratings.npz"
    np.savez_compressed(out, x=angles[trials].astype(np.float64), y=y, cell_ids=cells)
    print(f"  {len(cells)} cells x {n} trials -> {out.relative_to(ROOT)}")


def main(argv: list[str]) -> None:
    which = argv or ["nist", "fitbenchmarking", "sagefit"]
    for key in which:
        print(f"fetching {key}")
        if key == "nist":
            fetch_nist()
        elif key == "stringer2019":
            fetch_stringer2019()
        else:
            fetch_repo(key)


if __name__ == "__main__":
    main(sys.argv[1:])
