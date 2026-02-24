"""
🐉 BDH Explainer — Animated Interactive Pipeline Walkthrough
Inspired by Georgia Tech's Transformer Explainer.
Uses the REAL trained BDH model to show data flowing through each layer.

Architecture per layer (4 steps):
  1. RECALL:    x = ReLU(v* @ Dx)           — Sparse activation
  2. MECHANISM: a* = Attn(x, x, v*)         — Linear attention between cells
  3. EFFECT:    y = ReLU(LN(a*) @ Dy) * x   — Hebbian gate
  4. UPDATE:    v* = LN(v* + LN(y @ E))     — Residual update
"""
import streamlit as st
import streamlit.components.v1 as components
import torch
import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.bdh import (
    BDH, load_bdh, generate_board, format_board,
    compute_Gx, select_top_neurons, build_graph_edges, compute_force_layout,
    FLOOR, WALL, START, END, PATH, BOARD_NAMES
)
from core.analysis import per_layer_sparsity

st.set_page_config(page_title="BDH Explainer", page_icon="🐉", layout="wide")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "boardpath.pt")


@st.cache_resource
def load_model():
    model, bp_params, bdh_params = load_bdh(MODEL_PATH)
    model.eval()
    return model, bp_params, bdh_params


@st.cache_data
def get_topology(_model):
    """Extract graph topology from trained model weights (cached)."""
    Gx = compute_Gx(_model)
    selected, _ = select_top_neurons(_model, M=200, threshold=0.035)
    edges, weights, kept = build_graph_edges(Gx, selected, threshold=0.035,
                                              max_edges=1500, min_component_size=3)
    M = len(kept)
    positions = compute_force_layout(edges, np.abs(weights), M, seed=42)

    nodes_json = []
    for i in range(M):
        nodes_json.append({
            "id": i,
            "original_idx": int(kept[i]),
            "x": float(positions[i, 0]),
            "y": float(positions[i, 1]),
        })
    edges_json = []
    for (s, t), w in zip(edges, weights):
        edges_json.append({"source": int(s), "target": int(t), "weight": float(w)})
    return nodes_json, edges_json, kept


def run_inference(model, bp_params, bdh_params):
    """Run inference on a random board, return all frames."""
    board_size = bp_params.get("board_size", 10)
    input_board, target_board = generate_board(size=board_size, max_wall_prob=bp_params.get("wall_prob", 0.3))
    input_flat = input_board.flatten().unsqueeze(0)
    with torch.no_grad():
        logits, output_frames, x_frames, y_frames, attn_frames, logits_frames = model(input_flat, capture_frames=True)
        predicted = logits.argmax(dim=-1)
    return {
        "input_board": input_board.flatten().tolist(),
        "target_board": target_board.flatten().tolist(),
        "predicted": predicted[0].cpu().tolist(),
        "output_frames": output_frames,
        "x_frames": x_frames,
        "y_frames": y_frames,
        "attn_frames": attn_frames,
        "logits_frames": logits_frames,
        "board_size": board_size,
    }


def build_viz_json(inf_data, nodes_json, edges_json, kept_indices):
    """Build the JSON payload for the D3 visualization."""
    board_size = inf_data["board_size"]
    x_frames = inf_data["x_frames"]
    y_frames = inf_data["y_frames"]
    output_frames = inf_data["output_frames"]
    attn_frames = inf_data["attn_frames"]
    num_layers = len(x_frames)
    node_count = len(nodes_json)

    frames = []
    for l in range(num_layers):
        x_mean = x_frames[l].mean(dim=0).cpu().numpy()  # (N,)
        y_mean = y_frames[l].mean(dim=0).cpu().numpy()  # (N,)

        board_before = inf_data["input_board"] if l == 0 else output_frames[l - 1].cpu().tolist()
        board_after = output_frames[l].cpu().tolist()

        attn = attn_frames[l].cpu().numpy().tolist()
        if isinstance(attn[0][0], list):
            attn = attn[0]

        x_vals = [float(x_mean[int(kept_indices[i])]) for i in range(node_count)]
        y_vals = [float(y_mean[int(kept_indices[i])]) for i in range(node_count)]
        zeros = [0.0] * node_count

        # x sparsity for this layer
        x_active = float((x_frames[l] > 0).float().mean().item() * 100)
        y_active = float((y_frames[l] > 0).float().mean().item() * 100)

        # Per-cell activation for board overlay
        per_cell_x = (x_frames[l] > 0).float().mean(dim=1).cpu().numpy().tolist()  # (T,)

        # Step 1: Recall
        frames.append({
            "layer": l, "step_index": 0,
            "step_name": f"Layer {l} — 1. Recall",
            "description": f"Neurons read from residual stream (v*) and activate: x = ReLU(v* · Dx). Active: {x_active:.1f}%",
            "activations": x_vals, "prev_activations": zeros,
            "board_prediction": board_before, "attention": [],
            "x_active_pct": x_active, "y_active_pct": y_active,
            "per_cell_x": per_cell_x,
        })
        # Step 2: Mechanism (Attention)
        frames.append({
            "layer": l, "step_index": 1,
            "step_name": f"Layer {l} — 2. Mechanism",
            "description": f"Neurons attend to other tokens via linear attention. Blue arcs show where information flows between board cells.",
            "activations": x_vals, "prev_activations": zeros,
            "board_prediction": board_before, "attention": attn,
            "x_active_pct": x_active, "y_active_pct": y_active,
            "per_cell_x": per_cell_x,
        })
        # Step 3: Effect (Hebbian Gate)
        frames.append({
            "layer": l, "step_index": 2,
            "step_name": f"Layer {l} — 3. Effect",
            "description": f"Hebbian gate: y = ReLU(LN(a*) · Dy) ⊙ x. Only {y_active:.1f}% of synapses survive — fire together, wire together.",
            "activations": x_vals, "prev_activations": y_vals,
            "board_prediction": board_before, "attention": [],
            "x_active_pct": x_active, "y_active_pct": y_active,
            "per_cell_x": per_cell_x,
        })
        # Step 4: Update
        frames.append({
            "layer": l, "step_index": 3,
            "step_name": f"Layer {l} — 4. Update",
            "description": f"Residual stream updated: v* ← LN(v* + LN(y · E)). Board prediction refines.",
            "activations": x_vals, "prev_activations": y_vals,
            "board_prediction": board_after, "attention": [],
            "x_active_pct": x_active, "y_active_pct": y_active,
            "per_cell_x": per_cell_x,
        })

    return {
        "config": {
            "board_size": board_size,
            "num_layers": num_layers,
            "vocab_size": 5,
        },
        "topology": {"nodes": nodes_json, "edges": edges_json},
        "frames": frames,
        "input_board": inf_data["input_board"],
        "target_board": inf_data["target_board"],
        "final_prediction": inf_data["predicted"],
    }


# ─── Load & Run ───
model, bp_params, bdh_params = load_model()
nodes_json, edges_json, kept_indices = get_topology(model)

st.sidebar.header("⚙️ BDH Explainer")
if st.sidebar.button("🎲 Generate New Board", type="primary"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown(f"""
**Model:** {bdh_params.L} layers, {bdh_params.N} neurons, {bdh_params.H} heads
**Task:** Pathfinding on {bp_params.get('board_size', 10)}×{bp_params.get('board_size', 10)} boards
**Graph:** {len(nodes_json)} nodes, {len(edges_json)} edges
""")

inf_data = run_inference(model, bp_params, bdh_params)
viz_json = build_viz_json(inf_data, nodes_json, edges_json, kept_indices)
VIZ_DATA = json.dumps(viz_json)

# ─── The HTML/JS/D3 Visualization ───
html_code = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'Inter',sans-serif; background:#0e1117; color:#e0e0e0; overflow-x:hidden; }}

  /* ── Header ── */
  header {{
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border-bottom: 1px solid #0f3460;
    padding: 12px 20px;
    display: flex; align-items: center; justify-content: space-between;
  }}
  header h1 {{
    font-size: 1.3em;
    background: linear-gradient(90deg, #00d4ff, #e94560, #ffd700);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }}
  .chip {{ background:#0f3460; padding:3px 10px; border-radius:10px; font-size:0.7em; color:#00d4ff; margin-left:6px; }}

  /* ── Controls ── */
  .controls {{
    display: flex; align-items: center; gap: 12px;
    padding: 10px 20px; background: #12121a; border-bottom: 1px solid #1e1e2e;
  }}
  .controls button {{
    background: #0f3460; border: none; color: #00d4ff; padding: 8px 16px;
    border-radius: 8px; cursor: pointer; font-size: 0.85em; font-weight: 600;
    transition: all 0.2s;
  }}
  .controls button:hover {{ background: #1a5276; }}
  .controls button.play {{ background: #e94560; color: white; }}
  .controls button.play:hover {{ background: #c62828; }}
  .scrubber {{ flex: 1; accent-color: #00d4ff; }}

  /* ── Step pips ── */
  .step-pips {{ display: flex; align-items: center; gap: 4px; }}
  .pip {{ width: 10px; height: 10px; border-radius: 50%; background: #334155; transition: all 0.3s; }}
  .pip.active {{ background: #3b82f6; box-shadow: 0 0 8px #3b82f6; transform: scale(1.3); }}
  .pip.done {{ background: #22c55e; }}
  .pip-conn {{ width: 16px; height: 2px; background: #334155; transition: background 0.3s; }}
  .pip-conn.done {{ background: #22c55e; }}
  .pip-label {{ font-size: 9px; color: #666; text-align: center; }}

  /* ── Main ── */
  .main {{ display: flex; height: calc(100vh - 100px); overflow: hidden; }}

  /* ── Graph panel ── */
  .graph-panel {{ flex: 1; position: relative; background: #020617; }}
  .graph-panel svg {{ display: block; }}

  /* ── Info panel ── */
  .info-panel {{
    position: absolute; bottom: 12px; left: 12px; right: 320px;
    background: linear-gradient(135deg, rgba(30,41,59,0.95), rgba(15,23,42,0.95));
    backdrop-filter: blur(12px); border: 1px solid #334155;
    border-radius: 12px; padding: 14px 18px; z-index: 20;
  }}
  .info-panel .badge {{
    display: inline-block; padding: 2px 10px; border-radius: 999px;
    font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;
  }}
  .badge-recall {{ background: #1e3a5f; color: #60a5fa; }}
  .badge-mechanism {{ background: #312e81; color: #a78bfa; }}
  .badge-effect {{ background: #14532d; color: #4ade80; }}
  .badge-update {{ background: #7c2d12; color: #fb923c; }}
  .info-panel h3 {{ font-size: 1em; font-weight: 600; margin: 4px 0; color: white; }}
  .info-panel p {{ font-size: 0.85em; color: #94a3b8; line-height: 1.5; }}
  .stats-row {{ display: flex; gap: 12px; margin-top: 8px; }}
  .stats-row .stat {{ background: #1e293b; padding: 6px 12px; border-radius: 8px; text-align: center; }}
  .stats-row .stat .v {{ font-size: 1.2em; font-weight: 700; font-family: 'JetBrains Mono', monospace; }}
  .stats-row .stat .l {{ font-size: 0.65em; color: #64748b; }}

  /* ── Board panel ── */
  .board-panel {{
    width: 310px; border-left: 1px solid #1e293b; background: #0f172a;
    display: flex; flex-direction: column; overflow-y: auto;
  }}
  .board-panel h2 {{ font-size: 0.8em; color: #64748b; text-transform: uppercase; letter-spacing: 1px; padding: 12px 16px 8px; }}
  .board-grid {{
    display: grid; gap: 1px; background: #1e293b; margin: 0 12px;
    border-radius: 8px; overflow: hidden;
  }}
  .board-cell {{ aspect-ratio: 1; transition: background 0.3s, box-shadow 0.3s; position: relative; }}
  .board-cell .dot {{ position: absolute; top: 2px; right: 2px; width: 5px; height: 5px; border-radius: 50%; }}
  .attn-svg {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }}

  /* ── Legend ── */
  .legend {{ padding: 12px 16px; font-size: 0.75em; color: #94a3b8; }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; margin: 4px 0; }}
  .legend-swatch {{ width: 12px; height: 12px; border-radius: 3px; }}

  /* ── Equation bar ── */
  .eq-bar {{
    background: #1a1a2e; padding: 8px 16px; font-family: 'JetBrains Mono', monospace;
    font-size: 0.8em; color: #00d4ff; border-left: 3px solid #00d4ff;
    margin: 8px 12px; border-radius: 6px;
  }}
</style>
</head>
<body>
<div id="app">
  <header>
    <div style="display:flex;align-items:center;">
      <h1>🐉 BDH Explainer</h1>
      <span class="chip" id="layerChip">Layer 0</span>
    </div>
    <div style="font-size:0.75em;color:#64748b;">
      {bdh_params.L} layers · {bdh_params.N} neurons · {bdh_params.H} heads · Trained model
    </div>
  </header>

  <div class="controls">
    <div class="step-pips" id="stepPips">
      <div><div class="pip" data-s="0"></div><div class="pip-label">Recall</div></div>
      <div class="pip-conn" data-c="0"></div>
      <div><div class="pip" data-s="1"></div><div class="pip-label">Attend</div></div>
      <div class="pip-conn" data-c="1"></div>
      <div><div class="pip" data-s="2"></div><div class="pip-label">Effect</div></div>
      <div class="pip-conn" data-c="2"></div>
      <div><div class="pip" data-s="3"></div><div class="pip-label">Update</div></div>
    </div>
    <button id="btnPrev" title="Previous (←)">◀◀</button>
    <button id="btnPlay" class="play" title="Play/Pause (Space)">▶ Play</button>
    <button id="btnNext" title="Next (→)">▶▶</button>
    <input type="range" id="scrubber" class="scrubber" min="0" max="0" value="0">
    <span id="frameLabel" style="font-size:0.75em;color:#64748b;min-width:60px;text-align:right;">0 / 0</span>
  </div>

  <div class="main">
    <div class="graph-panel" id="graphContainer">
      <div class="info-panel" id="infoPanel">
        <span class="badge badge-recall" id="stepBadge">RECALL</span>
        <span style="font-size:0.7em;color:#475569;margin-left:8px;" id="stepCounter">Step 1/4</span>
        <h3 id="stepTitle">Layer 0 — Recall</h3>
        <p id="stepDesc">Loading...</p>
        <div class="stats-row" id="statsRow"></div>
      </div>
    </div>

    <div class="board-panel">
      <h2>Board State</h2>
      <div style="position:relative;margin:0 12px;">
        <div class="board-grid" id="boardGrid"></div>
        <svg class="attn-svg" id="attnSvg"></svg>
      </div>

      <div class="eq-bar" id="eqBar">x = ReLU(v* @ Dx)</div>

      <h2 style="margin-top:12px;">Target</h2>
      <div class="board-grid" id="targetGrid" style="margin:0 12px;"></div>

      <div class="legend">
        <div class="legend-item"><div class="legend-swatch" style="background:#16a34a;box-shadow:inset 0 0 4px #22c55e;"></div> Start (S)</div>
        <div class="legend-item"><div class="legend-swatch" style="background:#dc2626;box-shadow:inset 0 0 4px #ef4444;"></div> End (E)</div>
        <div class="legend-item"><div class="legend-swatch" style="background:#ca8a04;"></div> Path (*)</div>
        <div class="legend-item"><div class="legend-swatch" style="background:#334155;"></div> Wall (#)</div>
        <div class="legend-item"><div class="legend-swatch" style="background:#0f172a;"></div> Floor (.)</div>
        <hr style="border-color:#1e293b;margin:6px 0;">
        <div class="legend-item"><div style="width:12px;height:12px;border-radius:50%;background:#ef4444;"></div> Neuron activation (x)</div>
        <div class="legend-item"><div style="width:12px;height:12px;border-radius:50%;border:2px solid #3b82f6;"></div> Hebbian output (y)</div>
        <div class="legend-item"><div style="width:16px;height:2px;background:#60a5fa;"></div> Attention flow</div>
      </div>
      <div style="padding:8px 16px;font-size:0.7em;color:#475569;">
        <b>Controls:</b> ← → step · Space play/pause · Drag graph nodes
      </div>
    </div>
  </div>
</div>

<script>
const DATA = {VIZ_DATA};
const FRAMES = DATA.frames;
const BOARD_SIZE = DATA.config.board_size;
const COLORS = {{0:'#0f172a', 1:'#334155', 2:'#16a34a', 3:'#dc2626', 4:'#ca8a04'}};
const BADGE_CLASSES = ['badge-recall','badge-mechanism','badge-effect','badge-update'];
const BADGE_LABELS = ['RECALL','MECHANISM','EFFECT','UPDATE'];
const EQUATIONS = [
  'x = ReLU(v* @ Dx)  — encode to sparse neuron space',
  'a* = Attn(Q=x, K=x, V=v*)  — linear attention',
  'y = ReLU(LN(a*) @ Dy) ⊙ x  — Hebbian gate',
  'v* = LN(v* + LN(y @ E))  — residual update'
];

let currentFrame = 0;
let isPlaying = false;
let playInterval = null;
let simulation, svg, linkSel, nodeSel;

// ── Init Board ──
function initBoard(containerId, boardData) {{
  const grid = document.getElementById(containerId);
  grid.style.gridTemplateColumns = `repeat(${{BOARD_SIZE}}, 1fr)`;
  grid.innerHTML = '';
  for (let i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {{
    const cell = document.createElement('div');
    cell.className = 'board-cell';
    cell.style.backgroundColor = COLORS[boardData[i]] || COLORS[0];
    if (boardData[i] === 2) cell.style.boxShadow = 'inset 0 0 6px #22c55e';
    else if (boardData[i] === 3) cell.style.boxShadow = 'inset 0 0 6px #ef4444';
    grid.appendChild(cell);
  }}
}}

function updateBoard(frame) {{
  const grid = document.getElementById('boardGrid');
  const cells = grid.children;
  const pred = frame.board_prediction;
  const inp = DATA.input_board;
  const perCell = frame.per_cell_x || [];

  for (let i = 0; i < cells.length; i++) {{
    let val = pred[i];
    if (inp[i] === 2 || inp[i] === 3) val = inp[i];
    cells[i].style.backgroundColor = COLORS[val] || COLORS[0];
    cells[i].style.boxShadow = inp[i] === 2 ? 'inset 0 0 6px #22c55e' :
                                inp[i] === 3 ? 'inset 0 0 6px #ef4444' : 'none';
    // Activity dot
    let dot = cells[i].querySelector('.dot');
    if (perCell[i] > 0.05) {{
      if (!dot) {{ dot = document.createElement('div'); dot.className = 'dot'; cells[i].appendChild(dot); }}
      const alpha = Math.min(1, perCell[i] * 3);
      dot.style.backgroundColor = `rgba(239,68,68,${{alpha}})`;
    }} else if (dot) {{ dot.remove(); }}
  }}

  // Attention arcs
  const attnSvg = document.getElementById('attnSvg');
  attnSvg.innerHTML = '';
  if (frame.attention && frame.attention.length > 0) {{
    const attn = frame.attention;
    const T = attn.length;
    const rect = grid.getBoundingClientRect();
    const cw = rect.width / BOARD_SIZE, ch = rect.height / BOARD_SIZE;
    const flat = [];
    for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) if (i !== j) flat.push({{src:i, tgt:j, val:attn[i][j]}});
    flat.sort((a,b) => b.val - a.val);
    const topK = flat.slice(0, 20);
    const ns = 'http://www.w3.org/2000/svg';
    topK.forEach(e => {{
      if (e.val < 0.01) return;
      const x1 = (e.src % BOARD_SIZE + 0.5) * cw, y1 = (Math.floor(e.src / BOARD_SIZE) + 0.5) * ch;
      const x2 = (e.tgt % BOARD_SIZE + 0.5) * cw, y2 = (Math.floor(e.tgt / BOARD_SIZE) + 0.5) * ch;
      const line = document.createElementNS(ns, 'line');
      line.setAttribute('x1', x1); line.setAttribute('y1', y1);
      line.setAttribute('x2', x2); line.setAttribute('y2', y2);
      line.setAttribute('stroke', '#60a5fa');
      line.setAttribute('stroke-width', Math.max(1, e.val * 4));
      line.setAttribute('stroke-opacity', Math.min(0.8, e.val * 8));
      attnSvg.appendChild(line);
      const dot = document.createElementNS(ns, 'circle');
      dot.setAttribute('cx', x2); dot.setAttribute('cy', y2);
      dot.setAttribute('r', 2.5); dot.setAttribute('fill', '#60a5fa');
      dot.setAttribute('fill-opacity', Math.min(0.9, e.val * 8));
      attnSvg.appendChild(dot);
    }});
  }}
}}

// ── Init Graph ──
function initGraph() {{
  const container = document.getElementById('graphContainer');
  const w = container.clientWidth, h = container.clientHeight;
  const topo = DATA.topology;

  svg = d3.select(container).append('svg')
    .attr('width', '100%').attr('height', '100%')
    .attr('viewBox', [0, 0, w, h])
    .style('background-color', '#020617');

  const zoom = d3.zoom().scaleExtent([0.1, 8]).on('zoom', e => g.attr('transform', e.transform));
  svg.call(zoom);
  const g = svg.append('g');

  simulation = d3.forceSimulation(topo.nodes)
    .force('link', d3.forceLink(topo.edges).id(d => d.id).distance(50).strength(0.1))
    .force('charge', d3.forceManyBody().strength(-80))
    .force('center', d3.forceCenter(w / 2, h / 2))
    .force('x', d3.forceX(w / 2).strength(0.08))
    .force('y', d3.forceY(h / 2).strength(0.08));

  linkSel = g.append('g').attr('stroke', '#999').attr('stroke-opacity', 0.6)
    .selectAll('line').data(topo.edges).join('line')
    .attr('stroke-width', d => Math.sqrt(Math.abs(d.weight)) * 2.5);

  nodeSel = g.append('g').selectAll('.node').data(topo.nodes).join('g').attr('class', 'node')
    .call(d3.drag().on('start', (e,d) => {{ if(!e.active) simulation.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; }})
                    .on('drag', (e,d) => {{ d.fx=e.x; d.fy=e.y; }})
                    .on('end', (e,d) => {{ if(!e.active) simulation.alphaTarget(0); d.fx=null; d.fy=null; }}));

  nodeSel.append('circle').attr('class', 'ring').attr('r', 8).attr('fill', 'none')
    .attr('stroke', '#3b82f6').attr('stroke-width', 0);
  nodeSel.append('circle').attr('class', 'core').attr('r', 5).attr('fill', '#4b5563');
  nodeSel.append('title').text(d => `Neuron ${{d.original_idx}}`);

  simulation.on('tick', () => {{
    linkSel.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
           .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
    nodeSel.attr('transform', d => `translate(${{d.x}},${{d.y}})`);
  }});
}}

function updateGraph(frame) {{
  if (!nodeSel || !linkSel) return;
  const act = frame.activations, prev = frame.prev_activations;
  const colorScale = d3.interpolateRgb('#4b5563', '#ef4444');

  nodeSel.each(function(d, i) {{
    const g = d3.select(this);
    const xVal = act[i], yVal = prev[i];
    const t = Math.min(xVal / 2.0, 1.0);
    g.select('.core').attr('fill', colorScale(t)).attr('r', 5 + t * 5);
    const tY = Math.min(yVal / 2.0, 1.0);
    g.select('.ring').attr('stroke-width', tY * 3).attr('stroke-opacity', tY);
  }});

  linkSel.each(function(d) {{
    const s = d.source.index, t = d.target.index;
    const flow = Math.max(0, prev[s] * d.weight * act[t]);
    const ft = Math.min(flow * 15, 1);
    const el = d3.select(this);
    if (ft > 0.05) {{
      el.attr('stroke', '#cbd5e1').attr('stroke-opacity', 0.5 + 0.5*ft).attr('stroke-width', 0.5 + ft*2);
    }} else {{
      el.attr('stroke', '#334155').attr('stroke-opacity', 0.1).attr('stroke-width', 0.5);
    }}
  }});
}}

// ── UI Bindings ──
function setFrame(idx) {{
  idx = Math.max(0, Math.min(idx, FRAMES.length - 1));
  currentFrame = idx;
  const frame = FRAMES[idx];
  const step = frame.step_index;

  document.getElementById('layerChip').textContent = `Layer ${{frame.layer}}`;
  document.getElementById('scrubber').value = idx;
  document.getElementById('frameLabel').textContent = `${{idx + 1}} / ${{FRAMES.length}}`;

  // Pips
  document.querySelectorAll('.pip').forEach((p, i) => {{
    p.classList.remove('active', 'done');
    if (i === step) p.classList.add('active');
    else if (i < step) p.classList.add('done');
  }});
  document.querySelectorAll('.pip-conn').forEach((c, i) => {{
    c.classList.remove('done');
    if (i < step) c.classList.add('done');
  }});

  // Info panel
  const badge = document.getElementById('stepBadge');
  badge.textContent = BADGE_LABELS[step];
  badge.className = 'badge ' + BADGE_CLASSES[step];
  document.getElementById('stepCounter').textContent = `Step ${{step+1}}/4`;
  document.getElementById('stepTitle').textContent = frame.step_name;
  document.getElementById('stepDesc').textContent = frame.description;

  // Stats
  document.getElementById('statsRow').innerHTML = `
    <div class="stat"><div class="v" style="color:#ef4444;">${{frame.x_active_pct?.toFixed(1) || '—'}}%</div><div class="l">x active</div></div>
    <div class="stat"><div class="v" style="color:#3b82f6;">${{frame.y_active_pct?.toFixed(1) || '—'}}%</div><div class="l">y active</div></div>
    <div class="stat"><div class="v" style="color:#ffd700;">L${{frame.layer}}</div><div class="l">Layer</div></div>
  `;

  // Equation
  document.getElementById('eqBar').textContent = EQUATIONS[step];

  updateGraph(frame);
  updateBoard(frame);
}}

function togglePlay() {{
  if (isPlaying) {{ pause(); }} else {{ play(); }}
}}

function play() {{
  isPlaying = true;
  document.getElementById('btnPlay').textContent = '⏸ Pause';
  if (currentFrame >= FRAMES.length - 1) setFrame(0);
  playInterval = setInterval(() => {{
    if (currentFrame < FRAMES.length - 1) setFrame(currentFrame + 1);
    else pause();
  }}, 500);
}}

function pause() {{
  isPlaying = false;
  document.getElementById('btnPlay').textContent = '▶ Play';
  if (playInterval) {{ clearInterval(playInterval); playInterval = null; }}
}}

// Events
document.getElementById('btnPrev').addEventListener('click', () => {{ setFrame(currentFrame - 1); pause(); }});
document.getElementById('btnNext').addEventListener('click', () => {{ setFrame(currentFrame + 1); pause(); }});
document.getElementById('btnPlay').addEventListener('click', togglePlay);
document.getElementById('scrubber').max = FRAMES.length - 1;
document.getElementById('scrubber').addEventListener('input', e => {{ setFrame(parseInt(e.target.value)); pause(); }});
document.addEventListener('keydown', e => {{
  if (e.key === 'ArrowRight') {{ setFrame(currentFrame + 1); pause(); }}
  else if (e.key === 'ArrowLeft') {{ setFrame(currentFrame - 1); pause(); }}
  else if (e.key === ' ') {{ e.preventDefault(); togglePlay(); }}
}});

// Init
initBoard('boardGrid', DATA.input_board);
initBoard('targetGrid', DATA.target_board);
initGraph();
setFrame(0);
</script>
</body>
</html>
"""

components.html(html_code, height=750, scrolling=False)

# Below the interactive viz: show boards as text
with st.expander("📋 Board Details (Text)", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Input Board**")
        st.code(format_board(torch.tensor(inf_data["input_board"]), inf_data["board_size"]))
    with c2:
        st.markdown("**Target (Ground Truth)**")
        st.code(format_board(torch.tensor(inf_data["target_board"]), inf_data["board_size"]))
    with c3:
        st.markdown("**Model Prediction**")
        st.code(format_board(torch.tensor(inf_data["predicted"]), inf_data["board_size"]))

    # Accuracy
    target = inf_data["target_board"]
    pred = inf_data["predicted"]
    correct = sum(1 for a, b in zip(target, pred) if a == b)
    st.metric("Cell Accuracy", f"{correct}/{len(target)} ({correct/len(target)*100:.1f}%)")
