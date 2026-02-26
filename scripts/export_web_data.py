#!/usr/bin/env python3
"""
Generate web/data/viz_data.json for the rebuilt bdh-visualizer web UI.
Compatible with version2/web schema, with additional explainability fields.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from core.analysis import top_attention_edges
from core.bdh import (
    END,
    FLOOR,
    PATH,
    START,
    WALL,
    build_graph_edges,
    compute_Gx,
    compute_force_layout,
    generate_board,
    load_bdh,
    select_top_neurons,
)
from core.runtime import resolve_model_path


def _find_best_case(
    model,
    board_size: int,
    wall_prob: float,
    fixed_start: Tuple[int, int] | None,
    fixed_end: Tuple[int, int] | None,
    max_attempts: int,
    device: torch.device,
):
    best_acc = -1.0
    best = None
    for _ in range(max_attempts):
        input_board, target_board = generate_board(
            size=board_size,
            max_wall_prob=wall_prob,
            fixed_start=fixed_start,
            fixed_end=fixed_end,
        )
        input_flat = input_board.flatten().unsqueeze(0).to(device)
        with torch.no_grad():
            logits, output_frames, x_frames, y_frames, attn_frames, logits_frames = model(
                input_flat, capture_frames=True
            )
        pred = logits.argmax(dim=-1)[0]
        acc = (pred == target_board.flatten().to(device)).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best = (
                input_board,
                target_board,
                pred.cpu(),
                output_frames,
                x_frames,
                y_frames,
                attn_frames,
                logits_frames,
            )
        if acc >= 1.0:
            break
    return best, best_acc


def _build_concept_probe(
    kept_original_indices: np.ndarray,
    x_frames: List[torch.Tensor],
    input_board_flat: np.ndarray,
    target_board_flat: np.ndarray,
) -> Dict:
    masks = {
        "floor": input_board_flat == FLOOR,
        "wall": input_board_flat == WALL,
        "start": input_board_flat == START,
        "end": input_board_flat == END,
        "path_target": target_board_flat == PATH,
    }

    neurons = []
    concept_names = list(masks.keys())
    for local_id, orig in enumerate(kept_original_indices.tolist()):
        layer_scores = {name: [] for name in concept_names}
        for xf in x_frames:
            act = xf.cpu().numpy()[:, int(orig)]  # (T,)
            for name, m in masks.items():
                if np.any(m):
                    layer_scores[name].append(float(np.mean(act[m])))
                else:
                    layer_scores[name].append(0.0)
        scores = {k: float(np.mean(v)) for k, v in layer_scores.items()}
        total = float(sum(scores.values())) + 1e-8
        best_concept = max(scores, key=scores.get)
        purity = float(scores[best_concept] / total)
        neurons.append(
            {
                "id": int(local_id),
                "original_idx": int(orig),
                "scores": scores,
                "best_concept": best_concept,
                "purity": purity,
            }
        )

    concept_count = {k: 0 for k in concept_names}
    for n in neurons:
        concept_count[n["best_concept"]] += 1
    return {
        "neurons": neurons,
        "summary": concept_count,
    }


def export_data(
    output_path: str,
    m_neurons: int = 1000,
    threshold: float = 0.035,
    start_pos: Tuple[int, int] | None = None,
    end_pos: Tuple[int, int] | None = None,
    max_attempts: int = 50,
) -> Path:
    model_path = resolve_model_path()
    device = torch.device("cpu")
    model, boardpath_params, bdh_params = load_bdh(str(model_path), map_location=device)
    model.to(device)
    model.eval()

    board_size = int(boardpath_params.get("board_size", 16))
    wall_prob = float(boardpath_params.get("wall_prob", 0.3))
    best, best_acc = _find_best_case(
        model,
        board_size,
        wall_prob,
        start_pos,
        end_pos,
        max_attempts=max_attempts,
        device=device,
    )
    (
        input_board,
        target_board,
        predicted,
        output_frames,
        x_frames,
        y_frames,
        attn_frames,
        logits_frames,
    ) = best

    selected_indices, _ = select_top_neurons(model, M=m_neurons, threshold=threshold)
    gx = compute_Gx(model)
    edges, edge_weights, kept_original_indices = build_graph_edges(
        gx,
        selected_indices,
        threshold=threshold,
        max_edges=2000,
        min_component_size=3,
    )
    node_count = len(kept_original_indices)
    pos = compute_force_layout(edges, np.abs(edge_weights), node_count, seed=42)

    nodes_json = []
    for i in range(node_count):
        nodes_json.append(
            {
                "id": int(i),
                "original_idx": int(kept_original_indices[i]),
                "x": float(pos[i, 0]),
                "y": float(pos[i, 1]),
            }
        )

    edges_json = []
    for (src, tgt), w in zip(edges, edge_weights):
        edges_json.append(
            {
                "source": int(src),
                "target": int(tgt),
                "weight": float(w),
            }
        )

    frames_json = []
    num_layers = len(x_frames)
    input_pred = input_board.flatten().tolist()

    for l in range(num_layers):
        x_mean = x_frames[l].cpu().numpy().mean(axis=0)
        y_mean = y_frames[l].cpu().numpy().mean(axis=0)

        if l == 0:
            board_before = input_pred
        else:
            board_before = output_frames[l - 1].cpu().numpy().tolist()
        board_after = output_frames[l].cpu().numpy().tolist()

        attn = attn_frames[l]
        if attn.dim() == 3:
            attn = attn[0]
        attn = attn.cpu().numpy().tolist()

        x_vals = [float(x_mean[int(orig)]) for orig in kept_original_indices]
        y_vals = [float(y_mean[int(orig)]) for orig in kept_original_indices]
        zeros = [0.0] * node_count

        frames_json.append(
            {
                "layer": l,
                "activations": x_vals,
                "prev_activations": zeros,
                "board_prediction": board_before,
                "attention": [],
                "step_index": 0,
                "step_name": f"Layer {l} - 1. Recall",
                "description": "x = ReLU(v* @ Dx): sparse neuron recall from residual stream.",
            }
        )
        frames_json.append(
            {
                "layer": l,
                "activations": x_vals,
                "prev_activations": zeros,
                "board_prediction": board_before,
                "attention": attn,
                "step_index": 1,
                "step_name": f"Layer {l} - 2. Mechanism",
                "description": "Attention routes information between board positions.",
            }
        )
        frames_json.append(
            {
                "layer": l,
                "activations": x_vals,
                "prev_activations": y_vals,
                "board_prediction": board_before,
                "attention": [],
                "step_index": 2,
                "step_name": f"Layer {l} - 3. Effect",
                "description": "y = ReLU(LN(a*) @ Dy) * x: Hebbian-gated effect.",
            }
        )
        frames_json.append(
            {
                "layer": l,
                "activations": x_vals,
                "prev_activations": y_vals,
                "board_prediction": board_after,
                "attention": [],
                "step_index": 3,
                "step_name": f"Layer {l} - 4. Update",
                "description": "v* <- LN(v* + LN(y @ E)): residual update to next layer.",
            }
        )

    input_flat_np = input_board.flatten().cpu().numpy()
    target_flat_np = target_board.flatten().cpu().numpy()
    concept_probe = _build_concept_probe(
        kept_original_indices,
        x_frames,
        input_flat_np,
        target_flat_np,
    )

    attention_atlas = []
    for l, af in enumerate(attn_frames):
        attention_atlas.append(
            {
                "layer": int(l),
                "top_edges": top_attention_edges(af, top_k=24),
            }
        )

    export_obj = {
        "config": {
            "board_size": board_size,
            "num_layers": int(bdh_params.L),
            "vocab_size": int(bdh_params.V),
            "M_neurons": int(m_neurons),
            "model_path": str(model_path),
            "best_accuracy_pct": round(best_acc * 100.0, 2),
        },
        "topology": {
            "nodes": nodes_json,
            "edges": edges_json,
        },
        "frames": frames_json,
        "input_board": input_board.flatten().tolist(),
        "target_board": target_board.flatten().tolist(),
        "final_prediction": predicted.tolist(),
        "attention_atlas": attention_atlas,
        "concept_probe": concept_probe,
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
    parser.add_argument("--out", default="web/data/viz_data.json")
    parser.add_argument("--neurons", type=int, default=1000)
    parser.add_argument("--threshold", type=float, default=0.035)
    parser.add_argument("--start", type=str, default=None, help="row,col")
    parser.add_argument("--end", type=str, default=None, help="row,col")
    parser.add_argument("--max-attempts", type=int, default=50)
    args = parser.parse_args()

    out = export_data(
        output_path=args.out,
        m_neurons=args.neurons,
        threshold=args.threshold,
        start_pos=_parse_pos(args.start),
        end_pos=_parse_pos(args.end),
        max_attempts=args.max_attempts,
    )
    print(f"Exported: {out}")


if __name__ == "__main__":
    main()
