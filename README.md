# 🐉 BDH Visualizer — Baby Dragon Hatchling Interactive Explorer

**Making Post-Transformer AI Visible and Understandable**

An interactive visualization suite that reveals the inner workings of the [Baby Dragon Hatchling (BDH)](https://arxiv.org/abs/2509.26507) architecture — the first post-transformer frontier model with Hebbian memory, sparse activations, and scale-free graph topology.

> Built for **Path A: Visualization and Inner Worlds** of the Pathway Post-Transformer Frontier AI Challenge (KRITI High Prep)

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 What We Built

A five-part interactive visualization web application that makes BDH's unique architectural properties **visceral and understandable** to researchers, engineers, and non-experts alike. Our visualizer covers:

1. **🧠 Sparse Brain** — Activation density comparator showing BDH's ~5% sparsity vs transformer's dense activations
2. **🕸️ Graph Brain** — Interactive force-directed graph explorer of BDH's emergent scale-free neuron topology 
3. **💡 Memory Formation** — Hebbian learning animator showing synapses strengthening in real-time
4. **📐 Architecture Walkthrough** — Step-by-step forward pass tracer with BDH vs Transformer comparisons
5. **🐉 BDH Explainer** — Animated interactive pipeline walkthrough inspired by [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)

---

## 💡 Key Insight About BDH

This visualizer reveals **five architectural breakthroughs** that make BDH fundamentally different from transformers:

| Property | Transformer | BDH |
|---|---|---|
| **Structure** | Dense matrix layers | Scale-free graph of neurons |
| **Activation** | Nearly all neurons fire (~50-100%) | Only ~5% fire (sparse + positive) |
| **Memory** | KV-cache (grows with context) | Hebbian synapses (constant size) |
| **Attention** | O(T²) quadratic | O(T) linear |
| **Interpretability** | Black box | Graph structure is directly visualizable |

The most powerful insight: **BDH's Hebbian gate (`xy_sparse = x_sparse * y_sparse`)** is the mathematical embodiment of "neurons that fire together wire together." This single operation creates constant-size memory, monosemantic synapses, and inference-time learning — all properties transformers fundamentally lack.

---

## 🖥️ How to Run Locally

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd bdh-visualizer

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`. Navigate between pages using the sidebar.

---

## 🌐 Hosted Demo

**[Access the live demo here →](#)** *(Update with your deployment URL)*

Deployment options:
- **HuggingFace Spaces**: Upload this repo as a Streamlit Space
- **Streamlit Cloud**: Connect your GitHub repo at [share.streamlit.io](https://share.streamlit.io)

---

## 📁 Project Structure

```
bdh-visualizer/
├── app.py                        # Main landing page
├── pages/
│   ├── 1_🧠_Sparse_Brain.py      # Activation density comparator
│   ├── 2_🕸️_Graph_Brain.py       # Topology explorer
│   ├── 3_💡_Memory_Formation.py   # Hebbian learning animator
│   ├── 4_📐_Architecture.py       # Architecture walkthrough
│   └── 5_🐉_BDH_Explainer.py     # Animated interactive explainer
├── core/
│   ├── __init__.py
│   ├── bdh.py                    # BDH model (from official Pathway repo + extensions)
│   ├── transformer.py            # Simple transformer for comparison
│   └── analysis.py               # Sparsity, graph, Hebbian analysis utilities
├── .streamlit/
│   └── config.toml               # Theme and server configuration
├── requirements.txt
└── README.md
```

---

## 📊 Visualization Details

### 🧠 Sparse Brain (Activation Density Comparator)
- **Side-by-side heatmaps**: BDH (~5% active) vs Transformer (~50%+ active) for the same input
- **Activation distribution histograms**: BDH's dramatic sparsity vs Transformer's dense distribution
- **Per-layer sparsity bar chart**: How sparsity varies across layers
- **Per-token activity chart**: Track active neuron count as each character is processed
- **Configurable**: Choose layer, number of displayed neurons, color scale

### 🕸️ Graph Brain (Emergent Topology Explorer)
- **Interactive force-directed graph**: Neuron connectivity derived from encoder/decoder weight matrices
- **Degree distribution (log-log)**: Evidence of scale-free (power-law) topology
- **Connectivity heatmap**: Full neuron-to-neuron weight matrix
- **Multi-head comparison**: Each head develops unique topology
- **Network metrics**: Nodes, edges, average degree, clustering coefficient, components
- **Configurable**: Layout algorithm, head selection, edge threshold, node coloring (degree/strength/community)

### 💡 Memory Formation (Hebbian Learning Animator)
- **Synapse timeline**: Cumulative strength of top-K synapses as tokens are processed
- **Instantaneous activation heatmap**: Hebbian gate values per token × synapse
- **Sparsity per token chart**: How selective memory formation is
- **Top active synapses bar chart**: Which synapses accumulate most memory
- **3D cumulative strength surface**: Panoramic view of memory formation
- **Live training**: Train a model from scratch and watch sparsity evolve in real-time

### 📐 Architecture Walkthrough
- **Architecture diagram**: Visual representation of BDH's computation pipeline
- **Step-by-step forward pass**: Trace tensors through every stage with shape, sparsity, and heatmap at each step
- **BDH vs Transformer comparison table**: Nine-property head-to-head comparison
- **Memory scaling chart**: BDH constant vs Transformer linear memory growth
- **Complexity scaling chart**: O(T) vs O(T²) attention computation cost
- **Mathematical deep dive**: Full forward pass equations with LaTeX

### 🐉 BDH Explainer (Animated Interactive Walkthrough)
Inspired by Georgia Tech's [Transformer Explainer](https://poloclub.github.io/transformer-explainer/), this page provides a full animated pipeline walkthrough of BDH's forward pass:

- **Clickable token ribbon**: Select any input token to see its journey through the network — hover for instant stats (active neurons %, strongest neuron, Hebbian gate output)
- **7 expandable pipeline stages**: Embedding → Encoder (D→N) → Sparse ReLU → Rotary Attention → Hebbian Gate → Decoder → Output Probabilities — each with icon, math equation, description, and live data
- **Animated data flow**: Pulsing SVG arrows between stages with scroll-triggered fade-in animations
- **Neuron grids**: 256-neuron grid visualizations showing active (cyan) vs silent neurons for Sparse ReLU, and Hebbian-gated synapses (red) for the Hebbian gate
- **Canvas heatmaps**: Full tokens × sampled-neurons activation heatmaps for both x_sparse and xy_sparse
- **Live stats cards**: Active %, max activation, strongest neuron, probability scores — all updating per selected token
- **BDH vs Transformer comparison table**: Inline comparison of complexity, sparsity, memory, and position encoding
- **Per-layer activity bar chart**: Active neuron percentage across all layers with current layer highlighted
- **Top-10 output predictions**: Gradient bar chart of softmax probabilities for next-token prediction
- **Sidebar controls**: Input text, layer/head selector, animation speed

---

## 🎥 Demo Video

[Watch the demo video →](#) *(Update with YouTube/video URL)*

---

## 👥 Team Members and Contributions

| Member | Contribution |
|--------|-------------|
| *Name 1* | Sparse Brain visualization, analysis utilities |
| *Name 2* | Graph Brain topology explorer, network analysis |
| *Name 3* | Hebbian Learning animator, training visualization |
| *Name 4* | Architecture walkthrough, comparison charts, README |

---

## ⚠️ Limitations and Future Scope

### Current Limitations
- Uses a **compact demo model** (4 layers, 128 embed dim) for fast CPU inference; activation patterns are demonstrative but not from a fully trained model
- Graph topology is derived from **weight matrix correlations** as a proxy for BDH's true G_x = E @ D interaction graph
- No pre-trained weights loaded — uses randomly initialized models (properties like sparsity are architectural and hold regardless)
- Training visualization is limited to small models and short texts for interactive speed

### Future Scope
- **Load pre-trained checkpoints** from Pathway's official repository for fully trained model analysis
- **Monosemanticity dashboard**: Systematically identify synapses encoding specific concepts (currency, country, etc.)
- **Pathfinding visualization**: Port krychu/bdh maze-solving demo to web
- **Enhanced BDH Explainer**: Add auto-play animation mode that walks through stages automatically, WebGL-powered neuron particle effects
- **3D WebGL neuron graph**: Three.js-based immersive topology exploration
- **Long-context benchmark**: Compare BDH and Transformer at 50K+ tokens with live memory tracking
- **Cross-lingual synapse analysis**: Explore whether synapses activate consistently across languages

---

## 📝 References

- A. Kosowski, P. Uznański, J. Chorowski, Z. Stamirowska, M. Bartoszkiewicz. *[The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain](https://arxiv.org/abs/2509.26507)*, arXiv (2025).
- Official BDH code: [github.com/pathwaycom/bdh](https://github.com/pathwaycom/bdh)
- Pathway: [pathway.com](https://pathway.com/)
- Inspiration: [Transformer Explainer (Georgia Tech)](https://poloclub.github.io/transformer-explainer/)

---

## 📄 License

MIT License. BDH model code adapted from [Pathway Technology, Inc.](https://github.com/pathwaycom/bdh) under MIT License.
