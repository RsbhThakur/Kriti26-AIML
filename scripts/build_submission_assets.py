#!/usr/bin/env python3
"""
Generate reproducible visualization assets from the canonical `version2` model.

Outputs:
- version2/web/data/viz_data.json
- version2/web/data/hebbian_data.json
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path):
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main():
    root = Path(__file__).resolve().parents[2]
    v2 = root / "version2"
    model = v2 / "boardpath.pt"
    if not model.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {model}")

    run(
        [
            sys.executable,
            "utils/export_viz_data.py",
            "--model",
            str(model),
            "--out",
            "web/data/viz_data.json",
            "--neurons",
            "300",
            "--threshold",
            "0.035",
        ],
        cwd=v2,
    )

    run(
        [
            sys.executable,
            "utils/hebbian_viz.py",
            "--model",
            str(model),
            "--out",
            "web/data/hebbian_data.json",
            "--n-boards",
            "30",
            "--neurons",
            "1000",
            "--threshold",
            "0.035",
            "--max-edges",
            "2000",
        ],
        cwd=v2,
    )

    print("\nDone. Assets regenerated in version2/web/data.")


if __name__ == "__main__":
    main()

