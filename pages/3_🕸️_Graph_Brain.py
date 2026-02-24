"""
🕸️ Graph Brain — Visualize the emergent neural circuit (Gx = E @ Dx).
Uses the REAL trained model weights to reveal scale-free topology.
"""
import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.bdh import (
    load_bdh, compute_Gx, select_top_neurons,
    build_graph_edges, compute_force_layout,
)

st.set_page_config(page_title="Graph Brain", page_icon="🕸️", layout="wide")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "boardpath.pt")


@st.cache_resource
def load_model():
    model, bp_params, bdh_params = load_bdh(MODEL_PATH)
    model.eval()
    return model, bp_params, bdh_params


@st.cache_data
def build_graph(M, threshold):
    Gx = compute_Gx(model)
    selected, scores = select_top_neurons(model, M=M, threshold=threshold)
    edges, weights, kept = build_graph_edges(
        Gx, selected, threshold=threshold, max_edges=2000, min_component_size=3
    )
    positions = compute_force_layout(edges, np.abs(weights), len(kept), seed=42)

    # Degree distribution
    degree = np.zeros(len(kept))
    for s, t in edges:
        degree[s] += 1
        degree[t] += 1

    return {
        "kept": kept.tolist(),
        "scores": [float(scores[k]) for k in kept],
        "edges": [list(e) for e in edges],
        "weights": weights.tolist(),
        "positions": positions.tolist(),
        "degree": degree.tolist(),
        "M": len(kept),
        "num_edges": len(edges),
    }


model, bp_params, bdh_params = load_model()

st.sidebar.header("🕸️ Graph Brain")
M = st.sidebar.slider("Max neurons", 50, 500, 200, step=50)
threshold = st.sidebar.slider("Edge threshold", 0.01, 0.1, 0.035, step=0.005)

graph = build_graph(M, threshold)

st.title("🕸️ Graph Brain")
st.markdown("""
The BDH model's **causal circuit** emerges from trained weights: **Gx = E @ Dx**.
This matrix reveals which neurons can causally influence each other — without any
explicit graph being specified during training. The topology shows **scale-free** properties
with hub neurons that route information across the network.
""")

# ── Metrics ──
c1, c2, c3, c4 = st.columns(4)
c1.metric("Neurons in Graph", graph["M"])
c2.metric("Edges", f"{graph['num_edges']:,}")
avg_deg = np.mean(graph["degree"])
c3.metric("Avg Degree", f"{avg_deg:.1f}")
max_deg = max(graph["degree"]) if graph["degree"] else 0
c4.metric("Max Degree (Hub)", f"{int(max_deg)}")

# ── 1. Interactive Force Graph (Plotly) ──
st.subheader("Neural Circuit Graph")

pos = np.array(graph["positions"])
edges = graph["edges"]
weights = graph["weights"]
degree = np.array(graph["degree"])

# Build edge traces
edge_x, edge_y = [], []
for (s, t), w in zip(edges, weights):
    edge_x.extend([pos[s, 0], pos[t, 0], None])
    edge_y.extend([pos[s, 1], pos[t, 1], None])

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=edge_x, y=edge_y, mode="lines",
    line=dict(width=0.5, color="rgba(100,116,139,0.3)"),
    hoverinfo="none", name="Edges"
))

fig.add_trace(go.Scatter(
    x=pos[:, 0].tolist(), y=pos[:, 1].tolist(),
    mode="markers",
    marker=dict(
        size=(5 + degree * 1.5).clip(5, 40).tolist(),
        color=degree.tolist(),
        colorscale="YlOrRd",
        showscale=True,
        colorbar=dict(title="Degree"),
        line=dict(width=0.5, color="#1e293b"),
    ),
    text=[f"Neuron {graph['kept'][i]}<br>Degree: {int(degree[i])}<br>Score: {graph['scores'][i]:.3f}"
          for i in range(graph["M"])],
    hoverinfo="text",
    name="Neurons",
))

fig.update_layout(
    template="plotly_dark", height=600,
    paper_bgcolor="#0e1117", plot_bgcolor="#020617",
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    showlegend=False,
    title="Gx = E @ Dx — Emergent Neural Circuit",
)
st.plotly_chart(fig, use_container_width=True)

# ── 2. Degree Distribution ──
st.subheader("Degree Distribution")
col1, col2 = st.columns(2)

with col1:
    fig_deg = go.Figure(data=go.Histogram(
        x=degree.tolist(), nbinsx=30, marker_color="#ef4444"
    ))
    fig_deg.update_layout(
        xaxis_title="Degree", yaxis_title="Count",
        template="plotly_dark", height=350,
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        title="Degree Distribution (Linear Scale)",
    )
    st.plotly_chart(fig_deg, use_container_width=True)

with col2:
    # Log-log for scale-free check
    deg_vals, deg_counts = np.unique(degree[degree > 0].astype(int), return_counts=True)
    fig_log = go.Figure(data=go.Scatter(
        x=np.log10(deg_vals).tolist(),
        y=np.log10(deg_counts).tolist(),
        mode="markers+lines", marker=dict(size=8, color="#fbbf24"),
        line=dict(color="#fbbf24", width=2),
    ))
    fig_log.update_layout(
        xaxis_title="log₁₀(Degree)", yaxis_title="log₁₀(Count)",
        template="plotly_dark", height=350,
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        title="Log-Log Degree Plot (Scale-Free Check)",
    )
    st.plotly_chart(fig_log, use_container_width=True)

# ── 3. Hub Neurons Table ──
st.subheader("Top Hub Neurons")
sorted_idx = np.argsort(degree)[::-1][:20]
hub_data = []
for i in sorted_idx:
    hub_data.append({
        "Neuron ID": graph["kept"][i],
        "Degree": int(degree[i]),
        "Importance Score": f"{graph['scores'][i]:.4f}",
        "Graph Node": i,
    })
st.dataframe(hub_data, use_container_width=True)

# ── 4. Edge Weight Distribution ──
st.subheader("Edge Weight Distribution")
fig_ew = go.Figure(data=go.Histogram(
    x=[abs(w) for w in weights], nbinsx=50, marker_color="#3b82f6"
))
fig_ew.update_layout(
    xaxis_title="|Weight|", yaxis_title="Count",
    template="plotly_dark", height=300,
    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
    title="Absolute Edge Weight Distribution",
)
st.plotly_chart(fig_ew, use_container_width=True)

st.markdown("""
---
**About Gx**: The matrix Gx = E @ Dx captures the **causal influence** between neurons.
Because E maps from neuron-space back to token-space and Dx maps from token-space to neuron-space,
their product reveals which neurons can communicate through the residual stream. The resulting
graph is not designed — it **emerges** from training on the pathfinding task.
""")
