"""
Sparse Brain: animated sparsity and activation diagnostics.
"""
import os
import sys

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.analysis import compute_sparsity, per_cell_activation, per_layer_sparsity
from core.bdh import load_bdh, generate_board
from core.runtime import resolve_model_path

st.set_page_config(page_title="Sparse Brain", page_icon="🧠", layout="wide")
MODEL_PATH = str(resolve_model_path())


@st.cache_resource
def load_model():
    model, bp_params, bdh_params = load_bdh(MODEL_PATH)
    model.eval()
    return model, bp_params, bdh_params


model, bp_params, bdh_params = load_model()

st.sidebar.header("🧠 Sparse Brain")
st.sidebar.caption(f"Checkpoint: `{MODEL_PATH}`")
if st.sidebar.button("🎲 New Board", type="primary"):
    st.cache_data.clear()
    st.rerun()


@st.cache_data
def run_and_analyze():
    board_size = bp_params.get("board_size", 10)
    input_board, target_board = generate_board(
        size=board_size, max_wall_prob=bp_params.get("wall_prob", 0.3)
    )
    inp = input_board.flatten().unsqueeze(0)
    with torch.no_grad():
        logits, _, x_frames, y_frames, _, _ = model(inp, capture_frames=True)
    predicted = logits.argmax(dim=-1)[0].cpu().tolist()

    x_sp, y_sp = per_layer_sparsity(x_frames, y_frames)
    cell_act = per_cell_activation(x_frames, board_size)
    overall_sp = compute_sparsity(x_frames, y_frames)

    x_freq = [(xf > 0).float().mean(dim=0).cpu().numpy() for xf in x_frames]
    y_freq = [(yf > 0).float().mean(dim=0).cpu().numpy() for yf in y_frames]

    return {
        "input_board": input_board.flatten().tolist(),
        "target_board": target_board.flatten().tolist(),
        "predicted": predicted,
        "x_sparsity": x_sp,
        "y_sparsity": y_sp,
        "cell_act": cell_act,
        "overall": overall_sp,
        "x_freq": x_freq,
        "y_freq": y_freq,
        "board_size": board_size,
        "num_layers": len(x_frames),
        "N": bdh_params.N,
    }


data = run_and_analyze()
layers = list(range(data["num_layers"]))
board_size = data["board_size"]

st.title("🧠 Sparse Brain")
st.markdown(
    "Animated view of BDH sparsity: x activations are sparse, and y gets further pruned by Hebbian gating."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Avg x Active", f"{data['overall']['x_mean']:.1f}%")
c2.metric("Avg y Active", f"{data['overall']['y_mean']:.1f}%")
c3.metric("Total Neurons", f"{data['N']:,}")
c4.metric("Layers", str(data["num_layers"]))

st.subheader("Per-Layer Activation Rates (Animated Build-Up)")
fig = go.Figure(
    data=[
        go.Bar(x=layers, y=[data["x_sparsity"][0]] + [0] * (len(layers) - 1), name="x (Recall)", marker_color="#ef4444"),
        go.Bar(x=layers, y=[data["y_sparsity"][0]] + [0] * (len(layers) - 1), name="y (Hebbian)", marker_color="#3b82f6"),
    ],
    frames=[
        go.Frame(
            name=str(i),
            data=[
                go.Bar(y=[data["x_sparsity"][j] if j <= i else 0 for j in layers]),
                go.Bar(y=[data["y_sparsity"][j] if j <= i else 0 for j in layers]),
            ],
        )
        for i in layers
    ],
)
fig.update_layout(
    xaxis_title="Layer",
    yaxis_title="% Neurons Active",
    template="plotly_dark",
    barmode="group",
    height=390,
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
    updatemenus=[{
        "type": "buttons",
        "x": 0.01,
        "y": 1.18,
        "direction": "left",
        "buttons": [
            {"label": "▶ Play", "method": "animate", "args": [None, {"frame": {"duration": 280, "redraw": True}, "fromcurrent": True}]},
            {"label": "⏸ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]},
        ],
    }],
    sliders=[{
        "active": 0,
        "x": 0.12,
        "len": 0.86,
        "y": 1.16,
        "steps": [{"label": str(i), "method": "animate", "args": [[str(i)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}]} for i in layers],
    }],
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Per-Cell Activation Heatmap (Animated Across Layers)")
max_cell = max(max(layer) for layer in data["cell_act"])
heat_anim = go.Figure(
    data=[go.Heatmap(z=np.array(data["cell_act"][0]).reshape(board_size, board_size), colorscale="YlOrRd", zmin=0, zmax=max_cell)],
    frames=[
        go.Frame(
            name=str(i),
            data=[go.Heatmap(z=np.array(data["cell_act"][i]).reshape(board_size, board_size), colorscale="YlOrRd", zmin=0, zmax=max_cell)],
        )
        for i in layers
    ],
)
heat_anim.update_layout(
    template="plotly_dark",
    height=430,
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
    yaxis=dict(autorange="reversed"),
    updatemenus=[{
        "type": "buttons",
        "x": 0.01,
        "y": 1.18,
        "direction": "left",
        "buttons": [
            {"label": "▶ Play", "method": "animate", "args": [None, {"frame": {"duration": 300, "redraw": True}, "fromcurrent": True}]},
            {"label": "⏸ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]},
        ],
    }],
    sliders=[{
        "active": 0,
        "x": 0.12,
        "len": 0.86,
        "y": 1.16,
        "steps": [{"label": str(i), "method": "animate", "args": [[str(i)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}]} for i in layers],
    }],
)
st.plotly_chart(heat_anim, use_container_width=True)

st.subheader("Firing Distribution (Animated Histograms)")
tab1, tab2 = st.tabs(["x (Recall)", "y (Hebbian Gate)"])
with tab1:
    hx = go.Figure(
        data=[go.Histogram(x=data["x_freq"][0], nbinsx=50, marker_color="#ef4444")],
        frames=[go.Frame(name=str(i), data=[go.Histogram(x=data["x_freq"][i], nbinsx=50, marker_color="#ef4444")]) for i in layers],
    )
    hx.update_layout(
        template="plotly_dark",
        height=340,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        xaxis_title="Firing Frequency",
        yaxis_title="Neuron Count",
        updatemenus=[{
            "type": "buttons",
            "x": 0.01,
            "y": 1.18,
            "direction": "left",
            "buttons": [
                {"label": "▶ Play", "method": "animate", "args": [None, {"frame": {"duration": 280, "redraw": True}, "fromcurrent": True}]},
                {"label": "⏸ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]},
            ],
        }],
        sliders=[{
            "active": 0,
            "x": 0.12,
            "len": 0.86,
            "y": 1.16,
            "steps": [{"label": str(i), "method": "animate", "args": [[str(i)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}]} for i in layers],
        }],
    )
    st.plotly_chart(hx, use_container_width=True)
with tab2:
    hy = go.Figure(
        data=[go.Histogram(x=data["y_freq"][0], nbinsx=50, marker_color="#3b82f6")],
        frames=[go.Frame(name=str(i), data=[go.Histogram(x=data["y_freq"][i], nbinsx=50, marker_color="#3b82f6")]) for i in layers],
    )
    hy.update_layout(
        template="plotly_dark",
        height=340,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        xaxis_title="Firing Frequency",
        yaxis_title="Neuron Count",
        updatemenus=[{
            "type": "buttons",
            "x": 0.01,
            "y": 1.18,
            "direction": "left",
            "buttons": [
                {"label": "▶ Play", "method": "animate", "args": [None, {"frame": {"duration": 280, "redraw": True}, "fromcurrent": True}]},
                {"label": "⏸ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]},
            ],
        }],
        sliders=[{
            "active": 0,
            "x": 0.12,
            "len": 0.86,
            "y": 1.16,
            "steps": [{"label": str(i), "method": "animate", "args": [[str(i)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}]} for i in layers],
        }],
    )
    st.plotly_chart(hy, use_container_width=True)

st.subheader("Hebbian Pruning Ratio (Animated Cursor)")
ratios = [y / x if x > 0 else 0 for x, y in zip(data["x_sparsity"], data["y_sparsity"])]
ratio = go.Figure(
    data=[
        go.Scatter(x=layers, y=ratios, mode="lines+markers", marker=dict(size=9, color="#fbbf24"), line=dict(color="#fbbf24", width=3), name="y/x ratio"),
        go.Scatter(x=[0], y=[ratios[0]], mode="markers", marker=dict(size=18, color="#38bdf8", line=dict(width=2, color="#e2e8f0")), name="Current Layer"),
    ],
    frames=[
        go.Frame(
            name=str(i),
            data=[
                go.Scatter(x=layers, y=ratios, mode="lines+markers", marker=dict(size=9, color="#fbbf24"), line=dict(color="#fbbf24", width=3)),
                go.Scatter(x=[i], y=[ratios[i]], mode="markers", marker=dict(size=18, color="#38bdf8", line=dict(width=2, color="#e2e8f0"))),
            ],
        )
        for i in layers
    ],
)
ratio.update_layout(
    template="plotly_dark",
    height=330,
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
    xaxis_title="Layer",
    yaxis_title="y / x",
    updatemenus=[{
        "type": "buttons",
        "x": 0.01,
        "y": 1.18,
        "direction": "left",
        "buttons": [
            {"label": "▶ Play", "method": "animate", "args": [None, {"frame": {"duration": 280, "redraw": True}, "fromcurrent": True}]},
            {"label": "⏸ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]},
        ],
    }],
    sliders=[{
        "active": 0,
        "x": 0.12,
        "len": 0.86,
        "y": 1.16,
        "steps": [{"label": str(i), "method": "animate", "args": [[str(i)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}]} for i in layers],
    }],
)
st.plotly_chart(ratio, use_container_width=True)

