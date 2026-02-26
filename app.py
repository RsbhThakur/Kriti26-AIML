"""
🐉 BDH Visualizer — Baby Dragon Hatchling Interactive Explorer
Main landing page. Uses a REAL trained BDH model (pathfinding on 10×10 boards).
"""
import streamlit as st
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from core.runtime import resolve_model_path

st.set_page_config(
    page_title="BDH Visualizer",
    page_icon="🐉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem;
    }
    .main-header h1 {
        font-size: 3rem;
        background: linear-gradient(135deg, #00d4ff, #e94560, #ffd700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .main-header p { color: #888; font-size: 1.1rem; }
    .stat-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .stat-card .value {
        font-size: 2rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    .stat-card .label { color: #888; font-size: 0.85rem; margin-top: 0.25rem; }
    .feature-card {
        background: #13131d;
        border: 1px solid #1e1e2e;
        border-radius: 12px;
        padding: 1.5rem;
        height: 100%;
        transition: all 0.3s;
    }
    .feature-card:hover {
        border-color: #00d4ff;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.1);
    }
    .feature-icon { font-size: 2rem; margin-bottom: 0.75rem; }
    .feature-title { font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem; }
    .feature-desc { color: #aaa; font-size: 0.9rem; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🐉 BDH Visualizer</h1>
    <p>Explore the inner workings of a <strong>trained</strong> Baby Dragon Hatchling model</p>
    <p style="font-size: 0.85rem; color: #666;">
        Pathfinding on 10×10 boards · 12 layers · 2048 neurons · 4 heads · Real trained weights
    </p>
</div>
""", unsafe_allow_html=True)

# Key stats
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("""
    <div class="stat-card">
        <div class="value" style="color:#e94560;">~3-5%</div>
        <div class="label">y activation sparsity</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div class="stat-card">
        <div class="value" style="color:#00d4ff;">O(T)</div>
        <div class="label">Linear attention complexity</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown("""
    <div class="stat-card">
        <div class="value" style="color:#ffd700;">2048</div>
        <div class="label">Neurons (scale-free graph)</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown("""
    <div class="stat-card">
        <div class="value" style="color:#7dff7d;">12</div>
        <div class="label">Layers of reasoning</div>
    </div>""", unsafe_allow_html=True)

st.divider()

# Feature cards
st.markdown("## 🗺️ Explore the Visualizations")
st.info("Start with `🚀 Frontier Studio` in the sidebar for the complete, submission-ready superset demo.")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🐉</div>
        <div class="feature-title">BDH Explainer — Animated Pipeline Walkthrough</div>
        <div class="feature-desc">
            Inspired by Georgia Tech's Transformer Explainer. Watch a real trained BDH
            model solve pathfinding <b>layer by layer</b>: see neurons activate (Recall),
            attention flow between board cells (Mechanism), Hebbian gating (Effect),
            and the board prediction update — all animated with a D3.js force-directed
            neuron graph and interactive board view.
            <br><br><b>Key insight:</b> See reasoning emerge across 12 layers of computation
        </div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🧠</div>
        <div class="feature-title">Sparse Brain — Activation Density Explorer</div>
        <div class="feature-desc">
            Visualize the dramatic sparsity of BDH: x activations fire ~20% of neurons,
            while y (Hebbian-gated) fires only ~3-5%. See per-layer sparsity charts,
            per-cell activation heatmaps on the board, and neuron population histograms
            — all from the trained model's real inference.
            <br><br><b>Key insight:</b> Sparsity = Efficiency + Interpretability
        </div>
    </div>""", unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🕸️</div>
        <div class="feature-title">Graph Brain — Emergent Topology Explorer</div>
        <div class="feature-desc">
            Explore the learned causal circuit Gx = E @ Dx — the neuron-to-neuron
            connectivity graph that emerged from training. See hub neurons, scale-free
            degree distribution, and modular organization. The model was NOT hard-coded
            to have hubs — it discovered them from random initialization.
            <br><br><b>Key insight:</b> BDH self-organizes into a brain-like graph
        </div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">💡</div>
        <div class="feature-title">Memory Formation — Hebbian Learning Across Layers</div>
        <div class="feature-desc">
            Watch synapse strength accumulate across 12 layers as the model reasons
            about the pathfinding problem. Track the top-K most active synapses,
            see cumulative memory formation, and understand how y = ReLU(a* @ Dy) × x
            implements "neurons that fire together wire together."
            <br><br><b>Key insight:</b> Hebbian gating enables inference-time learning
        </div>
    </div>""", unsafe_allow_html=True)

st.divider()

# About
st.markdown("## 📚 About This Model")
st.markdown("""
This visualizer uses a **real trained BDH model** from the
[krychu/bdh](https://github.com/krychu/bdh) implementation:

| Parameter | Value |
|---|---|
| **Task** | Pathfinding on 10×10 boards (find shortest path from S to E) |
| **Vocabulary** | 5 tokens: Floor (.), Wall (#), Start (S), End (E), Path (*) |
| **Layers (L)** | 12 |
| **Neurons (N)** | 2048 |
| **Latent dim (D)** | 64 |
| **Heads (H)** | 4 |
| **Sequence length (T)** | 100 (10×10 board cells) |
| **Position encoding** | RoPE (Rotary) |
| **Training** | 100 epochs, 8000 training boards, AdamW |

The model reads a flattened board as input and predicts the same board with PATH cells
marked on the shortest route. Each of the 12 layers refines the prediction, with
attention flowing from START and END toward the path cells.
""")

# Sidebar
st.sidebar.markdown("### 🐉 BDH Visualizer")
st.sidebar.markdown("Trained model · Pathfinding · 12 layers")
st.sidebar.markdown("---")
st.sidebar.markdown("**Navigate** using the sidebar pages →")
st.sidebar.markdown("""
- 🐉 **BDH Explainer** — Animated walkthrough
- 🧠 **Sparse Brain** — Activation density
- 🕸️ **Graph Brain** — Topology explorer
- 💡 **Memory Formation** — Hebbian dynamics
- 🚀 **Frontier Studio** — Unified final submission hub
""")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "[Paper](https://arxiv.org/abs/2509.26507) · "
    "[Code](https://github.com/krychu/bdh) · "
    "[Pathway](https://pathway.com/)"
)
st.sidebar.markdown("---")
st.sidebar.caption(f"Checkpoint: `{resolve_model_path()}`")
