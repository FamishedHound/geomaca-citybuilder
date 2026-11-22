"""Check a downloaded archive against ``MANIFEST.csv`` (size and SHA-256 of every file).

Usage (from the folder that contains the package)::

    python -m chapter5_geomaca.checkpoints.verify --root <archive>
    python -m chapter5_geomaca.checkpoints.verify --root <archive> --role checkpoint

``<archive>`` is the folder that contains ``cb_runs_v4/``, ``cb_runs_v3/`` and
``lbf/``. The exit status is 0 when every listed file is present and intact.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

MANIFEST = Path(__file__).resolve().parent / "MANIFEST.csv"
ROLES = ("checkpoint", "run configuration", "LBF run record")


def read_manifest(path: Path = MANIFEST) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(root: Path, roles: Optional[Sequence[str]] = None) -> Dict[str, List[str]]:
    """Return the manifest paths that are missing, have the wrong size, or the wrong hash."""
    problems: Dict[str, List[str]] = {"missing": [], "size": [], "sha256": [], "ok": []}
    for row in read_manifest():
        if roles and row["role"] not in roles:
            continue
        p = Path(root) / row["path"]
        if not p.is_file():
            problems["missing"].append(row["path"])
        elif p.stat().st_size != int(row["size_bytes"]):
            problems["size"].append(row["path"])
        elif sha256(p) != row["sha256"]:
            problems["sha256"].append(row["path"])
        else:
            problems["ok"].append(row["path"])
    return problems


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="archive folder (cb_runs_v4/, cb_runs_v3/, lbf/)")
    ap.add_argument("--role", nargs="+", choices=ROLES, default=None,
                    help="check only these kinds of file")
    args = ap.parse_args(argv)
    res = verify(Path(args.root), args.role)
    print(f"ok: {len(res['ok'])}  missing: {len(res['missing'])}  "
          f"wrong size: {len(res['size'])}  wrong SHA-256: {len(res['sha256'])}")
    for kind in ("missing", "size", "sha256"):
        for p in res[kind]:
            print(f"  {kind}: {p}")
    sys.exit(0 if not (res["missing"] or res["size"] or res["sha256"]) else 1)


if __name__ == "__main__":
    main()
