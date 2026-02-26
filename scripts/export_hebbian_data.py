#!/usr/bin/env python3
"""
Generate web/data/hebbian_data.json for the rebuilt bdh-visualizer web UI.
Compatible with version2/web/js/hebbian.js.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from core.bdh import (
    build_graph_edges,
    compute_Gx,
    compute_force_layout,
    generate_board,
    load_bdh,
    select_top_neurons,
)
from core.runtime import resolve_model_path


def compute_hebbian(
    output_path: str = "web/data/hebbian_data.json",
    n_boards: int = 30,
    m_neurons: int = 1000,
    threshold: float = 0.035,
    max_edges: int = 2000,
    fixed_start: Tuple[int, int] | None = None,
    fixed_end: Tuple[int, int] | None = None,
):
    model_path = resolve_model_path()
    device = torch.device("cpu")
    model, bp_params, bdh_params = load_bdh(str(model_path), map_location=device)
    model.to(device)
    model.eval()
    board_size = int(bp_params.get("board_size", 16))
    wall_prob = float(bp_params.get("wall_prob", 0.3))

    selected_indices, _ = select_top_neurons(model, M=m_neurons, threshold=threshold)
    gx = compute_Gx(model)
    edges, edge_weights, kept_orig = build_graph_edges(
        gx,
        selected_indices,
        threshold=threshold,
        max_edges=max_edges,
        min_component_size=3,
    )
    node_count = len(kept_orig)
    positions = compute_force_layout(
        edges,
        np.abs(edge_weights),
        node_count,
        seed=42,
        iterations=200,
    )
    pmin, pmax = positions.min(axis=0), positions.max(axis=0)
    prng = np.maximum(pmax - pmin, 1e-6)
    pnorm = (positions - pmin) / prng

    nodes_json = [
        {
            "id": int(i),
            "original_idx": int(kept_orig[i]),
            "x": float(pnorm[i, 0]),
            "y": float(pnorm[i, 1]),
        }
        for i in range(node_count)
    ]
    edges_json = [
        {"source": int(s), "target": int(t), "weight": float(w)}
        for (s, t), w in zip(edges, edge_weights)
    ]

    edge_srcs = np.array([s for s, _ in edges], dtype=np.int32)
    edge_tgts = np.array([t for _, t in edges], dtype=np.int32)
    hebb_cum = np.zeros(len(edges), dtype=np.float64)
    correct_count = 0
    frames = []

    for board_idx in range(n_boards):
        input_board, target_board = generate_board(
            size=board_size,
            max_wall_prob=wall_prob,
            fixed_start=fixed_start,
            fixed_end=fixed_end,
        )
        input_flat = input_board.flatten().tolist()
        target_flat = target_board.flatten().tolist()
        x = input_board.flatten().unsqueeze(0).to(device)

        with torch.no_grad():
            logits, _, x_frames, _, _, _ = model(x, capture_frames=True)
            pred = logits.argmax(dim=-1)[0].cpu().tolist()

        if len(edges) > 0:
            for layer_idx in range(len(x_frames)):
                xf = x_frames[layer_idx].cpu().numpy()[:, kept_orig]
                svals = xf[:, edge_srcs]
                tvals = xf[:, edge_tgts]
                hebb_cum += (svals * tvals).sum(axis=0)

        path_found = any(v == 4 for v in pred)
        if path_found:
            correct_count += 1

        n_signal = sum(1 for v in target_flat if v != 0)
        n_match = sum(
            1 for i in range(len(pred)) if pred[i] == target_flat[i] and target_flat[i] != 0
        )
        acc = (n_match / n_signal * 100.0) if n_signal > 0 else 0.0

        pos = np.maximum(hebb_cum, 0.0)
        if len(pos) > 0 and np.max(pos) > 1e-8:
            p95 = np.percentile(pos, 95)
            h_norm = np.clip(pos / (p95 + 1e-8), 0.0, 1.0)
        else:
            h_norm = np.zeros_like(pos)

        node_strength = [0.0] * node_count
        for eidx, (s, t) in enumerate(edges):
            v = float(h_norm[eidx]) if eidx < len(h_norm) else 0.0
            node_strength[s] = max(node_strength[s], v)
            node_strength[t] = max(node_strength[t], v)

        frames.append(
            {
                "board_idx": int(board_idx),
                "board": input_flat,
                "target_board": target_flat,
                "predicted_board": pred,
                "path_found": bool(path_found),
                "path_cells": [i for i, v in enumerate(pred) if v == 4],
                "accuracy_pct": round(float(acc), 1),
                "hebbian_weights": [float(v) for v in h_norm.tolist()],
                "node_strength": node_strength,
                "h_raw_max": float(np.max(hebb_cum) if len(hebb_cum) > 0 else 0.0),
            }
        )

    export_obj = {
        "config": {
            "n_boards": int(n_boards),
            "board_size": int(board_size),
            "M_neurons": int(m_neurons),
            "n_edges": int(len(edges)),
            "num_layers": int(bdh_params.L),
            "model_path": str(model_path),
        },
        "topology": {
            "nodes": nodes_json,
            "edges": edges_json,
        },
        "frames": frames,
        "summary": {
            "correct_count": int(correct_count),
            "accuracy_pct": round(correct_count / max(n_boards, 1) * 100.0, 1),
        },
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(export_obj, f)
    return out


def _parse_pos(v: str | None):
    if not v:
        return None
    r, c = v.split(",")
    return int(r), int(c)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="web/data/hebbian_data.json")
    parser.add_argument("--n-boards", type=int, default=30)
    parser.add_argument("--neurons", type=int, default=1000)
    parser.add_argument("--threshold", type=float, default=0.035)
    parser.add_argument("--max-edges", type=int, default=2000)
    parser.add_argument("--start", type=str, default=None, help="row,col")
    parser.add_argument("--end", type=str, default=None, help="row,col")
    args = parser.parse_args()

    out = compute_hebbian(
        output_path=args.out,
        n_boards=args.n_boards,
        m_neurons=args.neurons,
        threshold=args.threshold,
        max_edges=args.max_edges,
        fixed_start=_parse_pos(args.start),
        fixed_end=_parse_pos(args.end),
    )
    print(f"Exported: {out}")


if __name__ == "__main__":
    main()
