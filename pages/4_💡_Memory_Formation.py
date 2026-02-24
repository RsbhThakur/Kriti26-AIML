"""
💡 Memory Formation — Track Hebbian synapse strength across layers.
Shows how the "fire together, wire together" principle drives BDH learning.
"""
import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.bdh import load_bdh, generate_board, format_board
from core.analysis import (
    hebbian_synapse_stats, cumulative_synapse_strength,
    top_attention_edges,
)

st.set_page_config(page_title="Memory Formation", page_icon="💡", layout="wide")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "boardpath.pt")


@st.cache_resource
def load_model():
    model, bp_params, bdh_params = load_bdh(MODEL_PATH)
    model.eval()
    return model, bp_params, bdh_params


model, bp_params, bdh_params = load_model()

st.sidebar.header("💡 Memory Formation")
if st.sidebar.button("🎲 New Board", type="primary"):
    st.cache_data.clear()
    st.rerun()


@st.cache_data
def run_analysis():
    board_size = bp_params.get("board_size", 10)
    input_board, target_board = generate_board(
        size=board_size, max_wall_prob=bp_params.get("wall_prob", 0.3)
    )
    inp = input_board.flatten().unsqueeze(0)
    with torch.no_grad():
        logits, out_frames, x_frames, y_frames, attn_frames, logit_frames = model(
            inp, capture_frames=True
        )
    predicted = logits.argmax(dim=-1)[0].cpu().tolist()

    hebb_stats = hebbian_synapse_stats(x_frames, y_frames)
    cum_strength = cumulative_synapse_strength(x_frames, y_frames)

    # Attention top edges per layer
    attn_per_layer = []
    for af in attn_frames:
        attn_per_layer.append(top_attention_edges(af, top_k=15))

    return {
        "input_board": input_board.flatten().tolist(),
        "target_board": target_board.flatten().tolist(),
        "predicted": predicted,
        "hebb_stats": hebb_stats,
        "cum_strength": cum_strength,
        "attn_per_layer": attn_per_layer,
        "board_size": board_size,
        "num_layers": len(x_frames),
    }


data = run_analysis()

st.title("💡 Memory Formation")
st.markdown("""
In BDH, memory forms through the **Hebbian gate**: `y = ReLU(LN(a*) @ Dy) * x`.
Two conditions must be met:
1. The neuron must **recall** (x > 0)
2. The attention output must be **positive** after projection

Only ~3-5% of synapses survive. This page tracks how these patterns strengthen across layers.
""")

# ── Metrics ──
hebb = data["hebb_stats"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Layers", data["num_layers"])
c2.metric("Avg Synapse Rate", f"{np.mean([h['y_active_pct'] for h in hebb]):.1f}%")
c3.metric("Avg Co-activation", f"{np.mean([h['coactivation'] for h in hebb]):.3f}")
c4.metric("Peak Hebbian Score",
          f"{max(h['hebbian_score'] for h in hebb):.4f}")

# ── 1. Hebbian Metrics Across Layers ──
st.subheader("Hebbian Metrics Across Layers")
layers = list(range(data["num_layers"]))

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=("x Active %", "y Active %", "Co-activation", "Hebbian Score"),
    vertical_spacing=0.15, horizontal_spacing=0.12,
)

x_pct = [h["x_active_pct"] for h in hebb]
y_pct = [h["y_active_pct"] for h in hebb]
coact = [h["coactivation"] for h in hebb]
hscore = [h["hebbian_score"] for h in hebb]

fig.add_trace(go.Bar(x=layers, y=x_pct, marker_color="#ef4444", name="x%"), row=1, col=1)
fig.add_trace(go.Bar(x=layers, y=y_pct, marker_color="#3b82f6", name="y%"), row=1, col=2)
fig.add_trace(go.Scatter(x=layers, y=coact, mode="lines+markers",
                          marker=dict(color="#fbbf24", size=8),
                          line=dict(color="#fbbf24", width=3), name="Co-act"), row=2, col=1)
fig.add_trace(go.Scatter(x=layers, y=hscore, mode="lines+markers",
                          marker=dict(color="#22c55e", size=8),
                          line=dict(color="#22c55e", width=3), name="Hebb"), row=2, col=2)
fig.update_layout(
    template="plotly_dark", height=500, showlegend=False,
    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
)
for i in range(1, 5):
    fig.update_xaxes(title_text="Layer", row=(i - 1) // 2 + 1, col=(i - 1) % 2 + 1)
st.plotly_chart(fig, use_container_width=True)

# ── 2. Cumulative Synapse Strength ──
st.subheader("Cumulative Synapse Strength Across Layers")
cum = data["cum_strength"]  # list of floats per layer (cumulative mean magnitude)

fig_cum = go.Figure()
fig_cum.add_trace(go.Scatter(
    x=layers, y=cum, mode="lines+markers+text",
    marker=dict(size=12, color="#a78bfa", symbol="diamond"),
    line=dict(color="#a78bfa", width=4),
    text=[f"{v:.3f}" for v in cum], textposition="top center",
    fill="tozeroy", fillcolor="rgba(167,139,250,0.1)",
))
fig_cum.update_layout(
    xaxis_title="Layer", yaxis_title="Cumulative Strength",
    template="plotly_dark", height=350,
    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
    title="How memories accumulate: synapse strength grows as the model builds its path solution",
)
st.plotly_chart(fig_cum, use_container_width=True)

# ── 3. Attention Flow per Layer ──
st.subheader("Top Attention Connections")
layer_pick = st.slider("Select Layer", 0, data["num_layers"] - 1, 0)

attn_edges = data["attn_per_layer"][layer_pick]
board_size = data["board_size"]

if attn_edges:
    fig_attn = go.Figure()
    # Draw board grid as background
    for r in range(board_size):
        for c in range(board_size):
            idx = r * board_size + c
            val = data["input_board"][idx]
            colors = {0: "#0f172a", 1: "#334155", 2: "#16a34a", 3: "#dc2626", 4: "#ca8a04"}
            fig_attn.add_shape(type="rect",
                               x0=c - 0.5, y0=r - 0.5, x1=c + 0.5, y1=r + 0.5,
                               fillcolor=colors.get(val, "#0f172a"),
                               line=dict(width=0.5, color="#1e293b"))

    # Draw attention arcs
    max_w = max(e["weight"] for e in attn_edges)
    for e in attn_edges:
        sr, sc = e["src"] // board_size, e["src"] % board_size
        tr, tc = e["tgt"] // board_size, e["tgt"] % board_size
        alpha = min(1.0, e["weight"] / max_w)
        fig_attn.add_annotation(
            x=tc, y=tr, ax=sc, ay=sr,
            arrowcolor=f"rgba(96,165,250,{alpha})",
            arrowwidth=1 + 3 * alpha, arrowhead=2,
            showarrow=True,
        )

    fig_attn.update_layout(
        template="plotly_dark", height=450,
        paper_bgcolor="#0e1117", plot_bgcolor="#020617",
        xaxis=dict(range=[-0.5, board_size - 0.5], showgrid=False, zeroline=False,
                   showticklabels=False, scaleanchor="y"),
        yaxis=dict(range=[board_size - 0.5, -0.5], showgrid=False, zeroline=False,
                   showticklabels=False),
        title=f"Layer {layer_pick} — Top {len(attn_edges)} Attention Flows",
    )
    st.plotly_chart(fig_attn, use_container_width=True)
else:
    st.info("No significant attention edges found for this layer.")

# ── 4. Board Comparison ──
st.subheader("Board Comparison")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**Input**")
    st.code(format_board(torch.tensor(data["input_board"]), data["board_size"]))
with c2:
    st.markdown("**Target**")
    st.code(format_board(torch.tensor(data["target_board"]), data["board_size"]))
with c3:
    st.markdown("**Predicted**")
    st.code(format_board(torch.tensor(data["predicted"]), data["board_size"]))
    correct = sum(1 for a, b in zip(data["target_board"], data["predicted"]) if a == b)
    T = len(data["target_board"])
    st.metric("Accuracy", f"{correct / T * 100:.1f}%")

st.markdown("""
---
**The Hebbian Principle in BDH:** "Neurons that fire together, wire together." The gate `y = ReLU(·) * x`
ensures that only neurons that are both recalled (x > 0) AND relevant (attention-projected output > 0)
contribute to the residual update. This natural selection at each layer is what makes BDH fundamentally
different from standard Transformers.
""")
