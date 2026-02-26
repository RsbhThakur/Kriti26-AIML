# BDH model — adapted from version2 (krychu/bdh)
# This is the REAL trained architecture for pathfinding on 10x10 boards.
# Checkpoint: model/boardpath.pt (trained for 100 epochs)

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import math
import torch.nn.functional as F
import torch
from torch import nn
import numpy as np
import random
from collections import deque


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

@dataclass
class BDHParameters:
    V: int    # vocabulary size
    T: int    # tokens (sequence length)
    H: int    # heads
    N: int    # neurons
    D: int    # latent dimension
    L: int    # layers
    dropout: float
    use_rope: bool
    use_abs_pos: bool


# ─────────────────────────────────────────────
# Board constants
# ─────────────────────────────────────────────

FLOOR = 0
WALL = 1
START = 2
END = 3
PATH = 4
VOCAB_SIZE = 5
BOARD_SYMBOLS = {FLOOR: '.', WALL: '#', START: 'S', END: 'E', PATH: '*'}
BOARD_NAMES = {FLOOR: 'Floor', WALL: 'Wall', START: 'Start', END: 'End', PATH: 'Path'}


# ─────────────────────────────────────────────
# RoPE
# ─────────────────────────────────────────────

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    Nh = x.shape[-1]
    x1 = x[..., :Nh // 2]
    x2 = x[..., Nh // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    q = q.to(cos.dtype)
    cos_ = cos.unsqueeze(0).unsqueeze(0)
    sin_ = sin.unsqueeze(0).unsqueeze(0)
    q = (q * cos_) + (rotate_half(q) * sin_)
    return q.to(q.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_T: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0
        self.head_dim = head_dim
        self.max_T = max_T
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(self.max_T, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len=None):
        if seq_len is None:
            seq_len = self.max_T
        assert seq_len <= self.max_T
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


class LinearAttention(nn.Module):
    def __init__(self, N: int, H: int, use_rope: bool, max_T: int):
        super().__init__()
        self.use_rope = use_rope
        if self.use_rope:
            self.rotary = RotaryEmbedding(head_dim=N // H, max_T=max_T, base=10000.0)

    def forward(self, Q, K, V, return_scores=False):
        if self.use_rope:
            _, _, T, _ = Q.size()
            cos, sin = self.rotary(T)
            QR = apply_rope(Q, cos, sin)
        else:
            QR = Q
        KR = QR
        scores = QR @ KR.mT
        output = scores @ V
        if return_scores:
            return output, scores
        return output


# ─────────────────────────────────────────────
# BDH Model
# ─────────────────────────────────────────────

class BDH(nn.Module):
    """
    Baby Dragon Hatchling — a biologically-inspired LLM architecture.

    Forward pass per layer:
      1. x = ReLU(v* @ Dx)        — Encode into sparse neuron space  (Recall)
      2. a* = Attn(x, x, v*)      — Linear attention in sparse space (Mechanism)
      3. y = ReLU(LN(a*) @ Dy) * x — Hebbian gate: fire-together-wire-together (Effect)
      4. v* = LN(v* + LN(y @ E))  — Decode back to residual stream   (Update)
    """

    def __init__(self, params: BDHParameters):
        super().__init__()
        V, T, H, N, D, L = params.V, params.T, params.H, params.N, params.D, params.L
        self.params = params
        self.N = N
        self.H = H
        self.L = L
        self.linear_attn = LinearAttention(N, H, params.use_rope, T)
        self.E = nn.Parameter(torch.zeros((N, D)).normal_(std=0.02))
        self.Dx = nn.Parameter(torch.zeros((H, D, N // H)).normal_(std=0.02))
        self.Dy = nn.Parameter(torch.zeros((H, D, N // H)).normal_(std=0.02))
        self.readout = nn.Parameter(torch.zeros((D, V)).normal_(std=0.02))
        self.emb = nn.Embedding(V, D)
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.drop = nn.Dropout(params.dropout)

        self.use_abs_pos = params.use_abs_pos
        if params.use_abs_pos:
            self.pos = nn.Embedding(T, D)
            self.register_buffer("pos_idx", torch.arange(T, dtype=torch.long), persistent=False)
            nn.init.normal_(self.pos.weight, mean=0.0, std=1.0 / math.sqrt(L))

    def forward(self, input_, capture_frames=False):
        B, T = input_.size()
        v_ast = self.ln(self.emb(input_).unsqueeze(1))  # B1TD

        if self.use_abs_pos:
            abs_pos_ast = self.pos(self.pos_idx)

        if capture_frames:
            output_frames: List[torch.Tensor] = []
            x_frames: List[torch.Tensor] = []
            y_frames: List[torch.Tensor] = []
            attn_frames: List[torch.Tensor] = []
            logits_frames: List[torch.Tensor] = []

        for _ in range(self.L):
            if self.use_abs_pos:
                v_ast = v_ast + abs_pos_ast

            # Step 1: RECALL — encode to sparse neuron space
            x = F.relu(v_ast @ self.Dx)   # BHTNh — sparse activations (~20% active)
            x = self.drop(x)

            # Step 2: MECHANISM — linear attention in sparse space
            if capture_frames:
                a_ast, attn_scores = self.linear_attn(x, x, v_ast, return_scores=True)
            else:
                a_ast = self.linear_attn(x, x, v_ast)

            # Step 3: EFFECT — Hebbian gate
            y = F.relu(self.ln(a_ast) @ self.Dy) * x  # BHTNh — y is VERY sparse (~3-5%)

            # Step 4: UPDATE — decode back to residual stream
            y_flat = y.transpose(1, 2).reshape(B, 1, T, self.N)
            y_flat = self.drop(y_flat)
            v_ast = v_ast + self.ln(y_flat @ self.E)
            v_ast = self.ln(v_ast)

            if capture_frames:
                self._capture_frame(v_ast, x, y_flat, T,
                                    output_frames, x_frames, y_frames)
                attn_frames.append(attn_scores.mean(dim=1).detach().clone())
                layer_logits = v_ast.squeeze(1) @ self.readout
                logits_frames.append(layer_logits.detach().clone())

        logits = v_ast.squeeze(1) @ self.readout  # BTV

        if capture_frames:
            return logits, output_frames, x_frames, y_frames, attn_frames, logits_frames
        return logits

    def _capture_frame(self, v_ast, x, y, T,
                       output_frames, x_frames, y_frames):
        logits = v_ast.squeeze(1) @ self.readout
        predicted = logits.argmax(dim=-1)
        output_frames.append(predicted[0])  # (T,)

        x_reshaped = x[0].transpose(0, 1).reshape(T, self.N)
        x_frames.append(x_reshaped.detach().clone())  # (T, N)

        y_frames.append(y[0, 0].detach().clone())  # (T, N)


# ─────────────────────────────────────────────
# Model I/O
# ─────────────────────────────────────────────

def load_bdh(path: str, map_location="cpu"):
    """Load trained BDH model from checkpoint.
    Returns: (model, boardpath_params_dict, bdh_params)
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    bdh_params = BDHParameters(**ckpt["bdh_params_dict"])
    model = BDH(bdh_params)
    model.load_state_dict(ckpt["bdh_state_dict"])
    boardpath_params = ckpt.get("boardpath_params_dict", {})
    return model, boardpath_params, bdh_params


# ─────────────────────────────────────────────
# Graph Topology (Gx = E @ Dx)
# ─────────────────────────────────────────────

def compute_Gx(model: BDH) -> np.ndarray:
    """Compute Gx = E @ Dx — the neuron-to-neuron causal circuit."""
    with torch.no_grad():
        H, D, Nh = model.Dx.shape
        N = H * Nh
        Dx_flat = model.Dx.permute(1, 0, 2).reshape(D, N)
        Gx = model.E @ Dx_flat  # (N, N)
        return Gx.detach().cpu().numpy()


def select_top_neurons(model: BDH, M: int = 300, threshold: float = 0.035):
    """Select top M neurons by degree in Gx.
    Returns: (selected_indices, index_map)
    """
    Gx = compute_Gx(model)
    abs_Gx = np.abs(Gx)
    np.fill_diagonal(abs_Gx, 0)
    edges_mask = abs_Gx > threshold
    score = edges_mask.sum(axis=0) + edges_mask.sum(axis=1)
    M = min(M, Gx.shape[0])
    selected = np.argsort(score)[-M:][::-1]
    index_map = {int(orig): new for new, orig in enumerate(selected)}
    return selected, index_map


def build_graph_edges(Gx, selected_indices, threshold=0.035, max_edges=2000, min_component_size=5):
    """Build edge list from Gx for selected neurons."""
    import networkx as nx

    Gx_sub = Gx[np.ix_(selected_indices, selected_indices)]
    M = len(selected_indices)
    abs_Gx = np.abs(Gx_sub)
    np.fill_diagonal(abs_Gx, 0)
    rows, cols = np.where(abs_Gx > threshold)
    edge_weights = Gx_sub[rows, cols]

    if len(rows) > max_edges:
        top_idx = np.argsort(np.abs(edge_weights))[-max_edges:]
        rows, cols = rows[top_idx], cols[top_idx]
        edge_weights = edge_weights[top_idx]

    G = nx.Graph()
    G.add_nodes_from(range(M))
    for r, c in zip(rows, cols):
        G.add_edge(int(r), int(c))

    components = list(nx.connected_components(G))
    large = [c for c in components if len(c) >= min_component_size]
    if not large and components:
        large = [max(components, key=len)]

    kept = set()
    for comp in large:
        kept.update(comp)
    kept_sorted = sorted(kept)
    old_to_new = {old: new for new, old in enumerate(kept_sorted)}

    final_edges, final_weights = [], []
    for idx, (r, c) in enumerate(zip(rows, cols)):
        if r in kept and c in kept:
            final_edges.append((old_to_new[r], old_to_new[c]))
            final_weights.append(edge_weights[idx])

    kept_original = selected_indices[kept_sorted]
    return final_edges, np.array(final_weights), kept_original


def compute_force_layout(edges, weights, M, seed=42, iterations=100):
    """Compute 2D force-directed layout using networkx."""
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(M))
    for idx, (src, tgt) in enumerate(edges):
        if idx < len(weights):
            G.add_edge(src, tgt, weight=weights[idx])
    pos_dict = nx.spring_layout(G, k=2.0 / np.sqrt(max(M, 1)),
                                iterations=iterations, seed=seed, weight='weight')
    positions = np.array([pos_dict[i] for i in range(M)])
    for dim in range(2):
        mn, mx = positions[:, dim].min(), positions[:, dim].max()
        if mx - mn > 1e-8:
            positions[:, dim] = 2 * (positions[:, dim] - mn) / (mx - mn) - 1
    return positions


# ─────────────────────────────────────────────
# Board Generation
# ─────────────────────────────────────────────

def generate_board(size=10, max_wall_prob=0.3, fixed_start=None, fixed_end=None):
    """Generate a valid board with shortest path (BFS).

    Args:
        size: board dimension.
        max_wall_prob: wall probability for non-fixed cells.
        fixed_start: optional tuple (r, c) for start position.
        fixed_end: optional tuple (r, c) for end position.
    """
    while True:
        board = [[FLOOR] * size for _ in range(size)]
        if fixed_start is not None:
            start = tuple(fixed_start)
            if not (0 <= start[0] < size and 0 <= start[1] < size):
                raise ValueError(f"fixed_start out of range: {fixed_start}")
        else:
            start = (random.randrange(size), random.randrange(size))

        if fixed_end is not None:
            end = tuple(fixed_end)
            if not (0 <= end[0] < size and 0 <= end[1] < size):
                raise ValueError(f"fixed_end out of range: {fixed_end}")
        else:
            end = (random.randrange(size), random.randrange(size))

        if end == start:
            if fixed_start is not None and fixed_end is not None:
                raise ValueError("fixed_start and fixed_end cannot be the same cell")
            while end == start:
                end = (random.randrange(size), random.randrange(size))
        board[start[0]][start[1]] = START
        board[end[0]][end[1]] = END
        for r in range(size):
            for c in range(size):
                if (r, c) not in (start, end) and random.random() < max_wall_prob:
                    board[r][c] = WALL
        prev = {start: None}
        q = deque([start])
        found = False
        while q:
            r, c = q.popleft()
            if (r, c) == end:
                found = True
                break
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size:
                    if board[nr][nc] != WALL and (nr, nc) not in prev:
                        prev[(nr, nc)] = (r, c)
                        q.append((nr, nc))
        if found:
            target = [row[:] for row in board]
            cur = end
            while cur != start:
                if cur != end:
                    target[cur[0]][cur[1]] = PATH
                cur = prev[cur]
            return torch.tensor(board, dtype=torch.long), torch.tensor(target, dtype=torch.long)


def format_board(board_tensor: torch.Tensor, board_size: int) -> str:
    board = board_tensor.view(board_size, board_size)
    return '\n'.join(
        ' '.join(BOARD_SYMBOLS.get(int(cell), str(int(cell))) for cell in row)
        for row in board
    )
