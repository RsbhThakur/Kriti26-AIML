"""
Memory Formation: animated Hebbian dynamics.
"""
import os
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.analysis import cumulative_synapse_strength, hebbian_synapse_stats, top_attention_edges
from core.bdh import format_board, generate_board, load_bdh
from core.runtime import resolve_model_path

st.set_page_config(page_title="Memory Formation", page_icon="💡", layout="wide")
MODEL_PATH = str(resolve_model_path())
MENU_STYLE = {
    "type": "buttons",
    "x": 0.01,
    "y": 1.18,
    "direction": "left",
    "bgcolor": "#1f2937",
    "bordercolor": "#4b5563",
    "font": {"color": "#f9fafb"},
    "buttons": [
        {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 320, "redraw": True}, "fromcurrent": True}]},
        {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]},
    ],
}


@st.cache_resource
def load_model():
    model, bp_params, bdh_params = load_bdh(MODEL_PATH)
    model.eval()
    return model, bp_params, bdh_params


model, bp_params, bdh_params = load_model()
st.sidebar.header("💡 Memory Formation")
st.sidebar.caption(f"Checkpoint: `{MODEL_PATH}`")
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
        logits, _, x_frames, y_frames, attn_frames, _ = model(inp, capture_frames=True)
    predicted = logits.argmax(dim=-1)[0].cpu().tolist()

    hebb_stats = hebbian_synapse_stats(x_frames, y_frames)
    cum_strength = cumulative_synapse_strength(x_frames, y_frames)
    attn_per_layer = [top_attention_edges(af, top_k=15) for af in attn_frames]
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
hebb = data["hebb_stats"]
layers = list(range(data["num_layers"]))

st.title("💡 Memory Formation")
st.markdown("Animated Hebbian memory evolution across layers and attention routing.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Layers", data["num_layers"])
c2.metric("Avg y Active", f"{np.mean([h['y_active_pct'] for h in hebb]):.1f}%")
c3.metric("Avg Co-activation", f"{np.mean([h['coactivation'] for h in hebb]):.3f}")
c4.metric("Peak Hebbian Score", f"{max(h['hebbian_score'] for h in hebb):.4f}")

st.subheader("Hebbian Metrics Across Layers (Animated Cursor)")
x_pct = [h["x_active_pct"] for h in hebb]
y_pct = [h["y_active_pct"] for h in hebb]
coact = [h["coactivation"] for h in hebb]
hscore = [h["hebbian_score"] for h in hebb]

fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=("x Active %", "y Active %", "Co-activation", "Hebbian Score"),
    vertical_spacing=0.15,
    horizontal_spacing=0.12,
)
fig.add_trace(go.Scatter(x=layers, y=x_pct, mode="lines+markers", line=dict(color="#ef4444"), name="x"), row=1, col=1)
fig.add_trace(go.Scatter(x=layers, y=y_pct, mode="lines+markers", line=dict(color="#3b82f6"), name="y"), row=1, col=2)
fig.add_trace(go.Scatter(x=layers, y=coact, mode="lines+markers", line=dict(color="#fbbf24"), name="co"), row=2, col=1)
fig.add_trace(go.Scatter(x=layers, y=hscore, mode="lines+markers", line=dict(color="#22c55e"), name="hebb"), row=2, col=2)
fig.add_trace(go.Scatter(x=[0], y=[x_pct[0]], mode="markers", marker=dict(size=14, color="#93c5fd"), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=[0], y=[y_pct[0]], mode="markers", marker=dict(size=14, color="#93c5fd"), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=[0], y=[coact[0]], mode="markers", marker=dict(size=14, color="#93c5fd"), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=[0], y=[hscore[0]], mode="markers", marker=dict(size=14, color="#93c5fd"), showlegend=False), row=2, col=2)
fig.frames = [
    go.Frame(
        name=str(i),
        data=[
            go.Scatter(x=layers, y=x_pct, mode="lines+markers"),
            go.Scatter(x=layers, y=y_pct, mode="lines+markers"),
            go.Scatter(x=layers, y=coact, mode="lines+markers"),
            go.Scatter(x=layers, y=hscore, mode="lines+markers"),
            go.Scatter(x=[i], y=[x_pct[i]], mode="markers"),
            go.Scatter(x=[i], y=[y_pct[i]], mode="markers"),
            go.Scatter(x=[i], y=[coact[i]], mode="markers"),
            go.Scatter(x=[i], y=[hscore[i]], mode="markers"),
        ],
    )
    for i in layers
]
fig.update_layout(
    template="plotly_dark",
    height=520,
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
    showlegend=False,
    updatemenus=[MENU_STYLE],
)
for i in range(1, 5):
    fig.update_xaxes(title_text="Layer", row=(i - 1) // 2 + 1, col=(i - 1) % 2 + 1)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Cumulative Synapse Strength (Animated Growth)")
cum = data["cum_strength"]
cum_fig = go.Figure(
    data=[go.Scatter(x=[0], y=[cum[0]], mode="lines+markers", marker=dict(size=11, color="#a78bfa"), line=dict(color="#a78bfa", width=4), fill="tozeroy", fillcolor="rgba(167,139,250,0.12)")],
    frames=[
        go.Frame(
            name=str(i),
            data=[go.Scatter(x=layers[: i + 1], y=cum[: i + 1], mode="lines+markers", marker=dict(size=11, color="#a78bfa"), line=dict(color="#a78bfa", width=4), fill="tozeroy", fillcolor="rgba(167,139,250,0.12)")],
        )
        for i in layers
    ],
)
cum_fig.update_layout(
    template="plotly_dark",
    height=340,
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
    xaxis_title="Layer",
    yaxis_title="Cumulative Strength",
    updatemenus=[MENU_STYLE],
)
st.plotly_chart(cum_fig, use_container_width=True)

st.subheader("Top Attention Connections (Animated by Layer)")
board_size = data["board_size"]
board = np.array(data["input_board"]).reshape(board_size, board_size)


def make_attention_traces(layer_idx: int):
    attn_edges = data["attn_per_layer"][layer_idx]
    line_x, line_y, tgt_x, tgt_y, tgt_s = [], [], [], [], []
    if attn_edges:
        max_w = max(e["weight"] for e in attn_edges) or 1.0
        for e in attn_edges:
            sr, sc = e["src"] // board_size, e["src"] % board_size
            tr, tc = e["tgt"] // board_size, e["tgt"] % board_size
            line_x.extend([sc, tc, None])
            line_y.extend([sr, tr, None])
            tgt_x.append(tc)
            tgt_y.append(tr)
            tgt_s.append(6 + 12 * (e["weight"] / max_w))

    heat = go.Heatmap(
        z=board,
        colorscale=[
            [0.00, "#0f172a"],
            [0.25, "#334155"],
            [0.50, "#16a34a"],
            [0.75, "#dc2626"],
            [1.00, "#ca8a04"],
        ],
        zmin=0,
        zmax=4,
        showscale=False,
        hoverinfo="skip",
    )
    lines = go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        line=dict(color="rgba(96,165,250,0.75)", width=2),
        hoverinfo="skip",
        showlegend=False,
    )
    targets = go.Scatter(
        x=tgt_x,
        y=tgt_y,
        mode="markers",
        marker=dict(size=tgt_s, color="rgba(125,211,252,0.85)", line=dict(color="#0ea5e9", width=1)),
        hoverinfo="skip",
        showlegend=False,
    )
    return [heat, lines, targets]


attn_anim = go.Figure(
    data=make_attention_traces(0),
    frames=[go.Frame(name=str(i), data=make_attention_traces(i)) for i in layers],
)
attn_anim.update_layout(
    template="plotly_dark",
    height=470,
    paper_bgcolor="#0e1117",
    plot_bgcolor="#020617",
    xaxis=dict(range=[-0.5, board_size - 0.5], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="y"),
    yaxis=dict(range=[board_size - 0.5, -0.5], showgrid=False, zeroline=False, showticklabels=False),
    title="Layer 0",
    updatemenus=[MENU_STYLE],
    sliders=[{
        "active": 0,
        "x": 0.12,
        "len": 0.86,
        "y": 1.16,
        "steps": [{"label": str(i), "method": "animate", "args": [[str(i)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}]} for i in layers],
    }],
)
st.plotly_chart(attn_anim, use_container_width=True)

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

