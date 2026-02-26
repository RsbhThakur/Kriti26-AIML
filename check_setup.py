#!/usr/bin/env python3
"""Sanity check for the submission app environment."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(__file__)
sys.path.insert(0, ROOT)


def check_import(name: str):
    try:
        mod = __import__(name)
        print(f"{name}: {getattr(mod, '__version__', 'ok')}")
    except Exception as exc:
        print(f"MISSING: {name} - {exc}")


def main():
    print("== Python Dependencies ==")
    for pkg in ["torch", "streamlit", "plotly", "networkx", "numpy", "scipy"]:
        check_import(pkg)

    print("\n== Model Load Test ==")
    try:
        from core.runtime import resolve_model_path
        from core.bdh import load_bdh, generate_board
        import torch

        ckpt = resolve_model_path()
        model, bp_params, bdh_params = load_bdh(str(ckpt))
        model.eval()
        print(f"checkpoint: {ckpt}")
        print(
            f"model: L={bdh_params.L}, N={bdh_params.N}, D={bdh_params.D}, H={bdh_params.H}, "
            f"board={bp_params.get('board_size', 10)}"
        )

        board, _ = generate_board(
            size=bp_params.get("board_size", 10),
            max_wall_prob=bp_params.get("wall_prob", 0.3),
        )
        x = board.flatten().unsqueeze(0)
        with torch.no_grad():
            logits, _, x_frames, y_frames, _, _ = model(x, capture_frames=True)
        print(f"inference: logits={tuple(logits.shape)}, layers={len(x_frames)}")
        print(f"sparsity sample: x={(x_frames[0] > 0).float().mean().item() * 100:.2f}% active")
        print(f"sparsity sample: y={(y_frames[0] > 0).float().mean().item() * 100:.2f}% active")
    except Exception as exc:
        print(f"MODEL ERROR: {exc}")
        raise

    print("\n== Setup looks good ==")


if __name__ == "__main__":
    main()

