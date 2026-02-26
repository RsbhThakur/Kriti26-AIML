"""
Graph Brain: animated topology explorer.
"""
import os
import sys

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.bdh import (
    build_graph_edges,
    compute_Gx,
    compute_force_layout,
    load_bdh,
    select_top_neurons,
)
from core.runtime import resolve_model_path

st.set_page_config(page_title="Graph Brain", page_icon="🕸️", layout="wide")
MODEL_PATH = str(resolve_model_path())


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


model, _, _ = load_model()
st.sidebar.header("🕸️ Graph Brain")
st.sidebar.caption(f"Checkpoint: `{MODEL_PATH}`")
M = st.sidebar.slider("Max neurons", 50, 600, 250, step=50)
threshold = st.sidebar.slider("Edge threshold", 0.01, 0.1, 0.035, step=0.005)
graph = build_graph(M, threshold)

st.title("🕸️ Graph Brain")
st.markdown("Animated reveal of `Gx = E @ Dx` connectivity by edge-strength percentiles.")

degree = np.array(graph["degree"])
weights = np.array(graph["weights"])
pos = np.array(graph["positions"])
edges = np.array(graph["edges"])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Neurons in Graph", graph["M"])
c2.metric("Edges", f"{graph['num_edges']:,}")
c3.metric("Avg Degree", f"{degree.mean():.1f}")
c4.metric("Max Hub Degree", f"{int(degree.max() if len(degree) else 0)}")

st.subheader("Neural Circuit Graph (Animated Edge Reveal)")
percentiles = [20, 35, 50, 65, 75, 85, 92, 97]
thresholds = [np.percentile(np.abs(weights), p) for p in percentiles] if len(weights) else [0.0]


def edge_trace_for(thr):
    mask = np.abs(weights) >= thr
    ex, ey = [], []
    for s, t in edges[mask]:
        ex.extend([pos[s, 0], pos[t, 0], None])
        ey.extend([pos[s, 1], pos[t, 1], None])
    return go.Scatter(
        x=ex,
        y=ey,
        mode="lines",
        line=dict(width=0.8, color="rgba(100,116,139,0.35)"),
        hoverinfo="none",
        name="Edges",
    )


fig_graph = go.Figure(
    data=[
        edge_trace_for(thresholds[0]),
        go.Scatter(
            x=pos[:, 0],
            y=pos[:, 1],
            mode="markers",
            marker=dict(
                size=(5 + degree * 1.3).clip(5, 38),
                color=degree,
                colorscale="YlOrRd",
                showscale=True,
                colorbar=dict(title="Degree"),
                line=dict(width=0.5, color="#1e293b"),
            ),
            text=[
                f"Neuron {graph['kept'][i]}<br>Degree: {int(degree[i])}<br>Score: {graph['scores'][i]:.3f}"
                for i in range(graph["M"])
            ],
            hoverinfo="text",
            name="Neurons",
        ),
    ],
    frames=[
        go.Frame(name=str(i), data=[edge_trace_for(thr), go.Scatter(x=pos[:, 0], y=pos[:, 1], mode="markers")])
        for i, thr in enumerate(thresholds)
    ],
)
fig_graph.update_layout(
    template="plotly_dark",
    height=620,
    paper_bgcolor="#0e1117",
    plot_bgcolor="#020617",
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    showlegend=False,
    title="Topology by Edge Strength Percentile",
    updatemenus=[{
        "type": "buttons",
        "x": 0.01,
        "y": 1.18,
        "direction": "left",
        "buttons": [
            {"label": "▶ Play", "method": "animate", "args": [None, {"frame": {"duration": 350, "redraw": True}, "fromcurrent": True}]},
            {"label": "⏸ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]},
        ],
    }],
    sliders=[{
        "active": 0,
        "x": 0.12,
        "len": 0.86,
        "y": 1.16,
        "steps": [
            {"label": f"P{percentiles[i]}", "method": "animate", "args": [[str(i)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}]}
            for i in range(len(percentiles))
        ],
    }],
)
st.plotly_chart(fig_graph, use_container_width=True)

st.subheader("Degree and Edge-Weight Distributions (Animated)")
col1, col2 = st.columns(2)

with col1:
    hubs = np.argsort(degree)[::-1]
    fig_deg = go.Figure(
        data=[go.Histogram(x=degree[hubs[:max(10, int(len(hubs) * 0.2))]], nbinsx=30, marker_color="#ef4444")],
        frames=[
            go.Frame(
                name=str(i),
                data=[go.Histogram(x=degree[hubs[: max(10, int(len(hubs) * frac))]], nbinsx=30, marker_color="#ef4444")],
            )
            for i, frac in enumerate(np.linspace(0.2, 1.0, 8))
        ],
    )
    fig_deg.update_layout(
        template="plotly_dark",
        height=350,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        title="Degree Histogram (Top-Hub Expansion)",
        xaxis_title="Degree",
        yaxis_title="Count",
        updatemenus=[{
            "type": "buttons",
            "x": 0.01,
            "y": 1.18,
            "direction": "left",
            "buttons": [
                {"label": "▶ Play", "method": "animate", "args": [None, {"frame": {"duration": 350, "redraw": True}, "fromcurrent": True}]},
                {"label": "⏸ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]},
            ],
        }],
    )
    st.plotly_chart(fig_deg, use_container_width=True)

with col2:
    abs_w = np.abs(weights)
    fig_w = go.Figure(
        data=[go.Histogram(x=abs_w, nbinsx=50, marker_color="#3b82f6")],
        frames=[
            go.Frame(
                name=str(i),
                data=[go.Histogram(x=abs_w[abs_w >= np.percentile(abs_w, p)], nbinsx=50, marker_color="#3b82f6")],
            )
            for i, p in enumerate([0, 20, 40, 55, 70, 82, 90, 95])
        ],
    )
    fig_w.update_layout(
        template="plotly_dark",
        height=350,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        title="Edge Weight Histogram (Progressive Thresholding)",
        xaxis_title="|Weight|",
        yaxis_title="Count",
        updatemenus=[{
            "type": "buttons",
            "x": 0.01,
            "y": 1.18,
            "direction": "left",
            "buttons": [
                {"label": "▶ Play", "method": "animate", "args": [None, {"frame": {"duration": 350, "redraw": True}, "fromcurrent": True}]},
                {"label": "⏸ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]},
            ],
        }],
    )
    st.plotly_chart(fig_w, use_container_width=True)

