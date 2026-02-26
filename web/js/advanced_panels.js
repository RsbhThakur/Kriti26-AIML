// advanced_panels.js
// Additional high-impact panels while preserving version2/web UI language.

const PANEL_MARGIN = { top: 26, right: 24, bottom: 42, left: 58 };

function axisStyle(sel) {
    sel.select(".domain").attr("stroke", "rgba(255,255,255,0.06)");
    sel.selectAll("line").attr("stroke", "rgba(255,255,255,0.05)");
    sel.selectAll("text")
        .attr("fill", "#a1a1aa")
        .attr("font-size", "11px")
        .attr("font-family", "'IBM Plex Mono', monospace");
}

function panelTitle(svg, text) {
    svg.append("text")
        .attr("x", PANEL_MARGIN.left)
        .attr("y", 18)
        .attr("fill", "#f4f4f5")
        .attr("font-size", 13)
        .attr("font-weight", 600)
        .text(text);
}

function getLayerAttentionFrames(data) {
    const out = [];
    const byLayer = new Map();
    for (const f of data.frames || []) {
        if (f.step_index === 1 && Array.isArray(f.attention) && f.attention.length) {
            byLayer.set(f.layer, f.attention);
        }
    }
    const layers = Array.from(byLayer.keys()).sort((a, b) => a - b);
    for (const l of layers) out.push({ layer: l, attn: byLayer.get(l) });
    return out;
}

export function renderAttentionAtlas(container, data) {
    container.innerHTML = "";
    const boardSize = data.config?.board_size || 16;
    const frames = getLayerAttentionFrames(data);
    if (!frames.length) {
        container.innerHTML = '<div class="hb-notice">No attention frames available.</div>';
        return;
    }

    const top = document.createElement("div");
    top.style.cssText = "display:flex;gap:0;min-height:0;flex:1;";
    const left = document.createElement("div");
    left.style.cssText = "flex:1;min-height:380px;";
    const right = document.createElement("div");
    right.style.cssText = "flex:0 0 360px;padding:10px 14px;overflow:auto;";
    top.appendChild(left);
    top.appendChild(right);
    container.appendChild(top);

    const controls = document.createElement("div");
    controls.style.cssText = "display:flex;align-items:center;gap:10px;padding:10px 12px;border:1px solid rgba(255,255,255,0.05);background:#0a0a0f;border-radius:10px;margin-top:10px;";
    controls.innerHTML = `
      <button id="attn-play" class="hb-btn hb-btn-play" style="min-width:90px;">Play</button>
      <input id="attn-layer" type="range" min="0" max="${frames.length - 1}" value="0" class="hb-range" style="width:220px;">
      <span id="attn-label" style="font-family:'IBM Plex Mono', monospace;color:#a1a1aa;font-size:11px;">Layer 0</span>
      <span id="attn-peak" style="font-family:'IBM Plex Mono', monospace;color:#fbbf24;font-size:11px;margin-left:auto;">Peak -</span>
    `;
    container.appendChild(controls);

    const W = left.clientWidth || 900;
    const H = 430;
    const svg = d3.select(left).append("svg").attr("width", W).attr("height", H);
    panelTitle(svg, "Attention Atlas: Incoming Influence per Board Cell");
    const g = svg
        .append("g")
        .attr("transform", `translate(${PANEL_MARGIN.left},${PANEL_MARGIN.top})`);
    const w = W - PANEL_MARGIN.left - PANEL_MARGIN.right;
    const h = H - PANEL_MARGIN.top - PANEL_MARGIN.bottom;

    const x = d3.scaleBand().domain(d3.range(boardSize)).range([0, w]).padding(0.02);
    const y = d3.scaleBand().domain(d3.range(boardSize)).range([0, h]).padding(0.02);
    const color = d3.scaleSequential(d3.interpolateTurbo).domain([0, 1]);

    const rows = d3.range(boardSize);
    const cols = d3.range(boardSize);
    const cells = [];
    for (const r of rows) for (const c of cols) cells.push({ r, c, idx: r * boardSize + c, value: 0 });
    const rects = g.selectAll("rect").data(cells).join("rect")
        .attr("x", d => x(d.c))
        .attr("y", d => y(d.r))
        .attr("width", x.bandwidth())
        .attr("height", y.bandwidth())
        .attr("rx", 2)
        .attr("fill", "#18181b");

    g.append("g").attr("transform", `translate(0,${h})`).call(d3.axisBottom(x).tickValues([])).call(axisStyle);
    g.append("g").call(d3.axisLeft(y).tickValues([])).call(axisStyle);

    const rightTitle = document.createElement("div");
    rightTitle.style.cssText = "font-size:12px;color:#e4e4e7;font-weight:600;margin-bottom:8px;";
    rightTitle.textContent = "Top Attention Connections";
    right.appendChild(rightTitle);
    const rightList = document.createElement("div");
    right.appendChild(rightList);

    let idx = 0;
    let timer = null;
    const setLayer = (k) => {
        idx = Math.max(0, Math.min(k, frames.length - 1));
        const attn = frames[idx].attn;
        const incoming = new Array(boardSize * boardSize).fill(0);
        for (let i = 0; i < attn.length; i++) {
            for (let j = 0; j < attn[i].length; j++) incoming[j] += attn[i][j];
        }
        const maxIn = Math.max(...incoming, 1e-8);
        rects
            .data(cells.map(c => ({ ...c, value: incoming[c.idx] / maxIn })))
            .transition()
            .duration(260)
            .attr("fill", d => color(d.value));

        const edges = (data.attention_atlas?.[idx]?.top_edges || []).slice(0, 12);
        rightList.innerHTML = edges.map((e, i2) => {
            const sr = Math.floor(e.src / boardSize);
            const sc = e.src % boardSize;
            const tr = Math.floor(e.tgt / boardSize);
            const tc = e.tgt % boardSize;
            return `<div style="display:flex;justify-content:space-between;font-family:'IBM Plex Mono', monospace;font-size:11px;color:#a1a1aa;padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.04);">
              <span>#${i2 + 1} (${sr},${sc}) -> (${tr},${tc})</span>
              <span style="color:#38bdf8">${(e.weight || 0).toFixed(3)}</span>
            </div>`;
        }).join("");

        document.getElementById("attn-label").textContent = `Layer ${frames[idx].layer}`;
        document.getElementById("attn-peak").textContent = `Peak ${(Math.max(...incoming)).toFixed(2)}`;
        document.getElementById("attn-layer").value = String(idx);
    };

    document.getElementById("attn-layer").addEventListener("input", (e) => setLayer(+e.target.value));
    document.getElementById("attn-play").addEventListener("click", (e) => {
        const btn = e.target;
        if (timer) {
            clearInterval(timer);
            timer = null;
            btn.textContent = "Play";
            return;
        }
        btn.textContent = "Pause";
        timer = setInterval(() => {
            idx = (idx + 1) % frames.length;
            setLayer(idx);
        }, 700);
    });

    setLayer(0);
}

export function renderConceptProbe(container, data) {
    container.innerHTML = "";
    const probe = data.concept_probe?.neurons || [];
    if (!probe.length) {
        container.innerHTML = '<div class="hb-notice">No concept probe data available.</div>';
        return;
    }

    const top = document.createElement("div");
    top.style.cssText = "display:flex;gap:0;min-height:0;flex:1;";
    const left = document.createElement("div");
    left.style.cssText = "flex:1;min-height:330px;";
    const right = document.createElement("div");
    right.style.cssText = "flex:0 0 420px;padding:8px 12px;overflow:auto;max-height:430px;";
    top.appendChild(left);
    top.appendChild(right);
    container.appendChild(top);

    const summary = data.concept_probe?.summary || {};
    const items = Object.entries(summary).map(([k, v]) => ({ concept: k, count: v }));

    const W = left.clientWidth || 900;
    const H = 390;
    const svg = d3.select(left).append("svg").attr("width", W).attr("height", H);
    panelTitle(svg, "Monosemantic Concept Probe (Best Concept per Neuron)");
    const g = svg.append("g").attr("transform", `translate(${PANEL_MARGIN.left},${PANEL_MARGIN.top + 8})`);
    const w = W - PANEL_MARGIN.left - PANEL_MARGIN.right;
    const h = H - PANEL_MARGIN.top - PANEL_MARGIN.bottom - 10;

    const x = d3.scaleBand().domain(items.map(d => d.concept)).range([0, w]).padding(0.24);
    const y = d3.scaleLinear().domain([0, d3.max(items, d => d.count) || 1]).nice().range([h, 0]);
    const c = d3.scaleOrdinal().domain(items.map(d => d.concept))
        .range(["#0ea5e9", "#f59e0b", "#10b981", "#f43f5e", "#8b5cf6"]);

    g.selectAll("rect").data(items).join("rect")
        .attr("x", d => x(d.concept))
        .attr("y", d => y(d.count))
        .attr("width", x.bandwidth())
        .attr("height", d => h - y(d.count))
        .attr("rx", 3)
        .attr("fill", d => c(d.concept))
        .attr("opacity", 0.78);

    g.append("g").attr("transform", `translate(0,${h})`).call(d3.axisBottom(x)).call(axisStyle);
    g.append("g").call(d3.axisLeft(y).ticks(5).tickFormat(d3.format("d"))).call(axisStyle);

    const sorted = [...probe].sort((a, b) => b.purity - a.purity).slice(0, 30);
    right.innerHTML = `
      <div style="font-size:12px;color:#e4e4e7;font-weight:600;margin-bottom:8px;">Top Neurons by Concept Purity</div>
      ${sorted.map((n, i) => `
        <div style="display:grid;grid-template-columns:40px 1fr 78px;gap:8px;align-items:center;padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.04);font-family:'IBM Plex Mono', monospace;font-size:11px;">
          <span style="color:#71717a">#${i + 1}</span>
          <span style="color:#a1a1aa">N${n.original_idx} -> <span style="color:#22d3ee">${n.best_concept}</span></span>
          <span style="color:#fbbf24;text-align:right;">${(n.purity * 100).toFixed(1)}%</span>
        </div>
      `).join("")}
    `;
}

export function renderScalingLab(container, data) {
    container.innerHTML = "";
    const card = document.createElement("div");
    card.style.cssText = "padding:12px;border:1px solid rgba(255,255,255,0.05);background:#0a0a0f;border-radius:10px;margin-bottom:10px;display:flex;gap:12px;align-items:center;";
    card.innerHTML = `
      <label style="font-size:11px;color:#a1a1aa;font-family:'IBM Plex Mono', monospace;">Context Length</label>
      <input id="scale-t" type="range" min="100" max="100000" step="100" value="5000" class="hb-range" style="width:260px;">
      <span id="scale-t-label" style="font-family:'IBM Plex Mono', monospace;color:#e4e4e7;font-size:12px;">5,000</span>
      <span id="scale-kv" style="margin-left:auto;font-family:'IBM Plex Mono', monospace;color:#f43f5e;font-size:11px;"></span>
      <span id="scale-hebb" style="font-family:'IBM Plex Mono', monospace;color:#10b981;font-size:11px;"></span>
    `;
    container.appendChild(card);

    const chart = document.createElement("div");
    chart.style.cssText = "min-height:360px;";
    container.appendChild(chart);

    const W = chart.clientWidth || 1000;
    const H = 390;
    const svg = d3.select(chart).append("svg").attr("width", W).attr("height", H);
    panelTitle(svg, "Scaling Lab: Transformer O(T^2) vs BDH O(T)");
    const g = svg.append("g").attr("transform", `translate(${PANEL_MARGIN.left},${PANEL_MARGIN.top + 8})`);
    const w = W - PANEL_MARGIN.left - PANEL_MARGIN.right;
    const h = H - PANEL_MARGIN.top - PANEL_MARGIN.bottom - 10;

    const ts = [100, 300, 600, 1000, 3000, 6000, 10000, 20000, 50000, 100000];
    const quad = ts.map(t => ({ t, v: t * t }));
    const lin = ts.map(t => ({ t, v: t }));

    const x = d3.scaleLog().domain([100, 100000]).range([0, w]);
    const y = d3.scaleLog().domain([100, 1e10]).range([h, 0]);

    g.append("g").attr("transform", `translate(0,${h})`)
        .call(d3.axisBottom(x).tickValues(ts).tickFormat(d3.format(",d"))).call(axisStyle);
    g.append("g")
        .call(d3.axisLeft(y).tickValues([1e2, 1e4, 1e6, 1e8, 1e10]).tickFormat(d3.format(".1e"))).call(axisStyle);

    const line = d3.line().x(d => x(d.t)).y(d => y(d.v)).curve(d3.curveMonotoneX);
    g.append("path").datum(quad).attr("fill", "none").attr("stroke", "#f43f5e").attr("stroke-width", 2.5).attr("d", line);
    g.append("path").datum(lin).attr("fill", "none").attr("stroke", "#10b981").attr("stroke-width", 2.5).attr("d", line);

    g.append("text").attr("x", x(50000)).attr("y", y(5e9)).attr("fill", "#f43f5e").attr("font-size", 11).text("Transformer KV/Attention");
    g.append("text").attr("x", x(50000)).attr("y", y(7e4)).attr("fill", "#10b981").attr("font-size", 11).text("BDH Hebbian state");

    const marker = g.append("line")
        .attr("y1", 0).attr("y2", h).attr("stroke", "#fbbf24")
        .attr("stroke-width", 1.2).attr("stroke-dasharray", "4,4");

    const updateT = (t) => {
        const kv = t * t;
        const hebb = (data.topology?.nodes?.length || data.config?.M_neurons || 1000) * (data.config?.num_layers || 12);
        document.getElementById("scale-t-label").textContent = Number(t).toLocaleString();
        document.getElementById("scale-kv").textContent = `Transformer ~ ${kv.toExponential(2)} units`;
        document.getElementById("scale-hebb").textContent = `BDH ~ ${hebb.toExponential(2)} units`;
        marker.attr("x1", x(t)).attr("x2", x(t));
    };

    const slider = document.getElementById("scale-t");
    slider.addEventListener("input", (e) => updateT(+e.target.value));
    updateT(+slider.value);
}
