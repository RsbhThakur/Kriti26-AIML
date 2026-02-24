"""
Analysis utilities for BDH visualization.
Works with the REAL trained model from version2.
"""
import numpy as np
import torch
from typing import List, Dict, Tuple


# ─────────────────────────────────────────────
# Sparsity Analysis
# ─────────────────────────────────────────────

def compute_sparsity(x_frames: List[torch.Tensor], y_frames: List[torch.Tensor]) -> dict:
    """Overall sparsity stats averaged across all layers.
    Returns dict with x_mean, y_mean (% active).
    """
    x_pcts = [(xf > 0).float().mean().item() * 100 for xf in x_frames]
    y_pcts = [(yf > 0).float().mean().item() * 100 for yf in y_frames]
    return {
        "x_mean": round(np.mean(x_pcts), 2),
        "y_mean": round(np.mean(y_pcts), 2),
    }


def compute_activation_stats(tensor: torch.Tensor) -> dict:
    """Detailed stats for a single activation tensor."""
    flat = tensor.flatten().float()
    total = flat.numel()
    active_mask = flat > 0
    active_count = active_mask.sum().item()
    stats = {
        "total_neurons": total,
        "active_neurons": active_count,
        "sparsity_pct": (1.0 - active_count / total) * 100 if total else 0,
        "active_pct": (active_count / total) * 100 if total else 0,
        "mean": flat.mean().item(),
        "std": flat.std().item(),
        "max": flat.max().item(),
    }
    if active_count > 0:
        active_vals = flat[active_mask]
        stats["active_mean"] = active_vals.mean().item()
        stats["active_std"] = active_vals.std().item()
    else:
        stats["active_mean"] = 0.0
        stats["active_std"] = 0.0
    return stats


def per_layer_sparsity(x_frames: List[torch.Tensor],
                       y_frames: List[torch.Tensor]) -> Tuple[List[float], List[float]]:
    """Compute activation % per layer for x and y.
    Returns: (x_pcts_list, y_pcts_list) — both are lists of floats.
    """
    x_pcts = [round((xf > 0).float().mean().item() * 100, 2) for xf in x_frames]
    y_pcts = [round((yf > 0).float().mean().item() * 100, 2) for yf in y_frames]
    return x_pcts, y_pcts


# ─────────────────────────────────────────────
# Per-cell (per-token) analysis
# ─────────────────────────────────────────────

def per_cell_activation(x_frames: List[torch.Tensor],
                        board_size: int) -> List[List[float]]:
    """Per-cell neuron activation for each layer.
    x_frames: list of (T, N) tensors (one per layer).
    Returns: list of lists — outer = layer, inner = per-token activation fraction.
    """
    result = []
    for xf in x_frames:
        per_token = (xf > 0).float().mean(dim=1).cpu().tolist()  # (T,)
        result.append(per_token)
    return result


def top_attention_edges(attn_frame: torch.Tensor, top_k: int = 15):
    """Extract top-K attention arcs from attention matrix.
    attn_frame: (B, T, T) or (T, T) — averaged over heads.
    Returns list of dicts with src, tgt, weight.
    """
    if attn_frame.dim() == 3:
        attn = attn_frame[0]  # first sample
    elif attn_frame.dim() == 2:
        attn = attn_frame
    else:
        return []
    T = attn.shape[0]
    attn = attn.clone().float()
    attn.fill_diagonal_(0)
    flat = attn.flatten()
    k = min(top_k, flat.numel())
    top_idx = torch.topk(flat, k).indices
    edges = []
    for idx in top_idx:
        i = idx.item() // T
        j = idx.item() % T
        val = attn[i, j].item()
        if val < 1e-4:
            continue
        edges.append({"src": i, "tgt": j, "weight": float(val)})
    return edges


# ─────────────────────────────────────────────
# Hebbian synapse analysis
# ─────────────────────────────────────────────

def hebbian_synapse_stats(x_frames: List[torch.Tensor],
                          y_frames: List[torch.Tensor]) -> List[dict]:
    """Per-layer Hebbian gate statistics.
    Returns a list of dicts (one per layer).
    """
    stats = []
    for xf, yf in zip(x_frames, y_frames):
        active_x = (xf > 0).float().mean().item() * 100
        active_y = (yf > 0).float().mean().item() * 100
        # Co-activation = fraction of neurons where both x>0 and y>0
        both_active = ((xf > 0) & (yf > 0)).float().mean().item()
        # Hebbian score = mean of y where y > 0
        y_active = yf[yf > 0]
        hebb_score = y_active.mean().item() if y_active.numel() > 0 else 0.0
        stats.append({
            "x_active_pct": round(active_x, 2),
            "y_active_pct": round(active_y, 2),
            "coactivation": round(both_active, 4),
            "hebbian_score": round(hebb_score, 4),
        })
    return stats


def cumulative_synapse_strength(x_frames: List[torch.Tensor],
                                y_frames: List[torch.Tensor]) -> List[float]:
    """Track cumulative synapse strength across layers.
    Returns a list of floats — one per layer (cumulative mean |y|).
    """
    N = y_frames[0].shape[1]
    cumulative = torch.zeros(N)
    result = []
    for yf in y_frames:
        layer_strength = yf.abs().mean(dim=0).cpu()  # (N,)
        cumulative = cumulative + layer_strength
        result.append(round(cumulative.mean().item(), 4))
    return result
