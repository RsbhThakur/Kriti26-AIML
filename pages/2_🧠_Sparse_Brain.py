"""
🧠 Sparse Brain — Visualize the ultra-sparse activations in BDH.
Uses the REAL trained model to show how only ~20% of neurons fire in x
and ~3-5% survive the Hebbian gate in y.
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
    per_layer_sparsity, per_cell_activation,
    compute_activation_stats, compute_sparsity,
)

st.set_page_config(page_title="Sparse Brain", page_icon="🧠", layout="wide")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "boardpath.pt")


@st.cache_resource
def load_model():
    model, bp_params, bdh_params = load_bdh(MODEL_PATH)
    model.eval()
    return model, bp_params, bdh_params


model, bp_params, bdh_params = load_model()

st.sidebar.header("🧠 Sparse Brain")
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
        logits, _, x_frames, y_frames, attn, logit_frames = model(inp, capture_frames=True)
    predicted = logits.argmax(dim=-1)[0].cpu().tolist()

    x_sp, y_sp = per_layer_sparsity(x_frames, y_frames)
    cell_act = per_cell_activation(x_frames, board_size)
    overall_sp = compute_sparsity(x_frames, y_frames)

    # Per-neuron activation frequency
    x_freq = []
    for xf in x_frames:
        x_freq.append((xf > 0).float().mean(dim=0).cpu().numpy())  # (N,)

    y_freq = []
    for yf in y_frames:
        y_freq.append((yf > 0).float().mean(dim=0).cpu().numpy())

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

st.title("🧠 Sparse Brain")
st.markdown("""
How the BDH brain keeps most neurons silent. In each layer, **x** (Recall) activates ~20% of neurons.
The **Hebbian gate y** further prunes this down to ~3-5%. This page lets you see exactly where activity flows.
""")

# ── Top metrics ──
c1, c2, c3, c4 = st.columns(4)
c1.metric("Avg x Active", f"{data['overall']['x_mean']:.1f}%")
c2.metric("Avg y Active", f"{data['overall']['y_mean']:.1f}%")
c3.metric("Total Neurons", f"{data['N']:,}")
c4.metric("Layers", str(data["num_layers"]))

# ── 1. Per-Layer Sparsity ──
st.subheader("Per-Layer Activation Rates")
fig = make_subplots(rows=1, cols=1)
layers = list(range(data["num_layers"]))
fig.add_trace(go.Bar(
    x=layers, y=data["x_sparsity"],
    name="x (Recall)", marker_color="#ef4444",
    text=[f"{v:.1f}%" for v in data["x_sparsity"]], textposition="outside"
))
fig.add_trace(go.Bar(
    x=layers, y=data["y_sparsity"],
    name="y (Hebbian)", marker_color="#3b82f6",
    text=[f"{v:.1f}%" for v in data["y_sparsity"]], textposition="outside"
))
fig.update_layout(
    xaxis_title="Layer", yaxis_title="% Neurons Active",
    template="plotly_dark", barmode="group", height=400,
    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
    font_color="#e0e0e0",
)
st.plotly_chart(fig, use_container_width=True)

# ── 2. Per-Cell Board Heatmap ──
st.subheader("Per-Cell Activation Heatmap")
layer_choice = st.slider("Layer", 0, data["num_layers"] - 1, 0)

col1, col2 = st.columns(2)
with col1:
    board_size = data["board_size"]
    cell_mat = np.array(data["cell_act"][layer_choice]).reshape(board_size, board_size)
    fig_heat = go.Figure(data=go.Heatmap(
        z=cell_mat, colorscale="YlOrRd", showscale=True,
        colorbar=dict(title="Activation"),
    ))
    fig_heat.update_layout(
        title=f"Neuron Activity per Board Cell (Layer {layer_choice})",
        template="plotly_dark", height=400,
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        yaxis=dict(autorange="reversed"), xaxis_title="Column", yaxis_title="Row",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

with col2:
    # Show the actual board
    inp_mat = np.array(data["input_board"]).reshape(board_size, board_size)
    NAMES = {0: ".", 1: "#", 2: "S", 3: "E", 4: "*"}
    board_text = [[NAMES.get(int(inp_mat[r][c]), "?") for c in range(board_size)] for r in range(board_size)]
    colors = {0: "#0f172a", 1: "#334155", 2: "#16a34a", 3: "#dc2626", 4: "#ca8a04"}
    board_colors = [[colors.get(int(inp_mat[r][c]), "#0f172a") for c in range(board_size)] for r in range(board_size)]
    fig_board = go.Figure(data=go.Heatmap(
        z=inp_mat, text=board_text, texttemplate="%{text}",
        colorscale=[[0, "#0f172a"], [0.25, "#334155"], [0.5, "#16a34a"], [0.75, "#dc2626"], [1, "#ca8a04"]],
        showscale=False
    ))
    fig_board.update_layout(
        title="Input Board", template="plotly_dark", height=400,
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        yaxis=dict(autorange="reversed"), xaxis_title="Column", yaxis_title="Row",
    )
    st.plotly_chart(fig_board, use_container_width=True)

# ── 3. Neuron Population Histogram ──
st.subheader("Neuron Firing Frequency Distribution")
tab1, tab2 = st.tabs(["x (Recall)", "y (Hebbian Gate)"])

with tab1:
    x_f = data["x_freq"][layer_choice]
    fig_hist = go.Figure(data=go.Histogram(x=x_f, nbinsx=50, marker_color="#ef4444"))
    fig_hist.update_layout(
        xaxis_title="Firing Frequency (across tokens)",
        yaxis_title="Number of Neurons",
        template="plotly_dark", height=350,
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        title=f"x (Recall) — Layer {layer_choice}: {(x_f > 0).sum()} / {len(x_f)} active",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with tab2:
    y_f = data["y_freq"][layer_choice]
    fig_hist2 = go.Figure(data=go.Histogram(x=y_f, nbinsx=50, marker_color="#3b82f6"))
    fig_hist2.update_layout(
        xaxis_title="Firing Frequency (across tokens)",
        yaxis_title="Number of Neurons",
        template="plotly_dark", height=350,
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        title=f"y (Hebbian) — Layer {layer_choice}: {(y_f > 0).sum()} / {len(y_f)} active",
    )
    st.plotly_chart(fig_hist2, use_container_width=True)

# ── 4. Sparsity Decay Across Layers ──
st.subheader("Sparsity Ratio Across Layers")
ratios = [y / x if x > 0 else 0 for x, y in zip(data["x_sparsity"], data["y_sparsity"])]
fig_ratio = go.Figure()
fig_ratio.add_trace(go.Scatter(
    x=layers, y=ratios, mode="lines+markers",
    marker=dict(size=10, color="#fbbf24"), line=dict(color="#fbbf24", width=3),
    name="y/x ratio", text=[f"{r:.2f}" for r in ratios], textposition="top center"
))
fig_ratio.update_layout(
    xaxis_title="Layer", yaxis_title="y / x Ratio",
    template="plotly_dark", height=350,
    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
    title="Hebbian Pruning Ratio (lower = more selective)",
)
st.plotly_chart(fig_ratio, use_container_width=True)
