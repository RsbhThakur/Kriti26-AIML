"""
Runtime helpers for model/checkpoint resolution.

Default priority:
1) BDH_MODEL_PATH env var
2) ../version2/boardpath.pt   (latest training workspace)
3) ./model/boardpath.pt       (packaged fallback)
"""
from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_model_path() -> Path:
    env_path = os.getenv("BDH_MODEL_PATH")
    candidates = []
    if env_path:
        candidates.append(Path(env_path).expanduser())

    root = project_root()
    candidates.extend(
        [
            root.parent / "version2" / "boardpath.pt",
            root / "model" / "boardpath.pt",
        ]
    )

    for path in candidates:
        if path.exists():
            return path

    searched = "\n".join(f" - {p}" for p in candidates)
    raise FileNotFoundError(
        "Unable to locate BDH checkpoint. Looked in:\n"
        f"{searched}\n\nSet BDH_MODEL_PATH=/absolute/path/to/boardpath.pt"
    )

