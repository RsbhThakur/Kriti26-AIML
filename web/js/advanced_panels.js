// advanced_panels.js
// High-impact analysis panels: Attention Atlas, Concept Probe, Scaling Lab
// Professional styling — matches version2/web design language

const PM = { top: 32, right: 24, bottom: 46, left: 58 };

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
        .attr("x", PM.left)
        .attr("y", 18)
        .attr("fill", "#f4f4f5")
        .attr("font-size", 13)
        .attr("font-weight", 600)
        .attr("font-family", "'Inter', sans-serif")
        .text(text);
}

function gridLines(g, yScale, w, ticks = 4) {
    g.append("g").attr("class", "grid")
        .call(d3.axisLeft(yScale).ticks(ticks).tickSize(-w).tickFormat(""))
        .call(gg => {
            gg.select(".domain").remove();
            gg.selectAll("line").attr("stroke", "rgba(255,255,255,0.03)").attr("stroke-dasharray", "2,4");
        });
}

function noDataNotice(container, msg) {
    container.innerHTML = `
        <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:300px;gap:12px;color:#52525b;">
            <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line>
            </svg>
            <p style="font-size:13px;font-family:'Inter',sans-serif;max-width:380px;text-align:center;line-height:1.6;">${msg}</p>
        </div>`;
}

// ── Helper: get attention frames from data ─────────────────────────────────────
function getAttentionFrames(data) {
    const out = [];
    const byLayer = new Map();
    for (const f of data.frames || []) {
        if (Array.isArray(f.attention) && f.attention.length) {
            if (!byLayer.has(f.layer)) byLayer.set(f.layer, f.attention);
        }
    }
    const layers = Array.from(byLayer.keys()).sort((a, b) => a - b);
    for (const l of layers) out.push({ layer: l, attn: byLayer.get(l) });
    return out;
}

// ── Helper: generate synthetic attention-like heatmap from board predictions ──
function syntheticAttentionHeatmap(data, boardSize) {
    const frames = data.frames || [];
    const result = [];
    const byLayer = new Map();
    for (const f of frames) {
        if (!byLayer.has(f.layer)) byLayer.set(f.layer, []);
        byLayer.get(f.layer).push(f);
    }
    const layers = Array.from(byLayer.keys()).sort((a, b) => a - b);
    const T = boardSize * boardSize;
    for (const l of layers) {
        const lf = byLayer.get(l);
        // Use the effect step (step_index=2) if available
        const f = lf.find(x => x.step_index === 2) || lf[0];
        // Create a T×T pseudo-attention from board and activations
        const acts = f.activations.slice(0, T);
        const incoming = new Float32Array(T);
        for (let i = 0; i < T; i++) {
            // Use normalized activation to represent "attention received"
            incoming[i] = (acts[i] || 0);
        }
        // Add path cell emphasis
        if (f.board_prediction) {
            for (let i = 0; i < Math.min(T, f.board_prediction.length); i++) {
                if (f.board_prediction[i] === 4) incoming[i] *= 2.5;
            }
        }
        result.push({ layer: l, incoming });
    }
    return result;
}

// ══════════════════════════════════════════════════════════════════════════════
// ATTENTION ATLAS
// ══════════════════════════════════════════════════════════════════════════════
export function renderAttentionAtlas(container, data) {
    container.innerHTML = "";
    const boardSize = data.config?.board_size || 16;
    const T = boardSize * boardSize;

    // Try real attention data first, fall back to synthetic
    const attnFrames = getAttentionFrames(data);
    const isSynthetic = attnFrames.length === 0;
    const synthFrames = isSynthetic ? syntheticAttentionHeatmap(data, boardSize) : null;
    const frameCount  = isSynthetic ? synthFrames.length : attnFrames.length;

    if (frameCount === 0) {
        noDataNotice(container, "No attention or activation data available. Run the model with <code>export_viz_data.py</code> to generate attention frames.");
        return;
    }

    // ── Notice if using synthetic data ──
    if (isSynthetic) {
        const note = document.createElement("div");
        note.style.cssText = "padding:8px 12px;background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.2);border-radius:8px;margin-bottom:10px;font-size:11px;color:#fbbf24;font-family:'IBM Plex Mono',monospace;";
        note.textContent = "ℹ Showing activation heatmap (proxy for attention) — add attention data to viz_data.json for full attention atlas.";
        container.appendChild(note);
    }

    // ── Layout ──
    const wrap = document.createElement("div");
    wrap.style.cssText = "display:flex;gap:16px;min-height:0;flex:1;";

    const leftSide = document.createElement("div");
    leftSide.style.cssText = "flex:1;min-height:380px;";

    const rightSide = document.createElement("div");
    rightSide.style.cssText = "flex:0 0 320px;display:flex;flex-direction:column;gap:10px;";

    wrap.appendChild(leftSide);
    wrap.appendChild(rightSide);
    container.appendChild(wrap);

    // ── Controls ──
    const ctrlRow = document.createElement("div");
    ctrlRow.style.cssText = "display:flex;align-items:center;gap:10px;padding:10px 12px;border:1px solid rgba(255,255,255,0.06);background:rgba(9,9,11,0.6);border-radius:10px;margin-top:10px;flex-wrap:wrap;";
    ctrlRow.innerHTML = `
      <button id="attn-play" style="padding:4px 14px;background:#6366f1;color:white;border:none;border-radius:6px;font-size:11px;font-weight:600;cursor:pointer;font-family:'IBM Plex Mono',monospace;min-width:70px;">▶ Play</button>
      <input id="attn-layer" type="range" min="0" max="${frameCount - 1}" value="0" style="flex:1;min-width:120px;cursor:pointer;">
      <span id="attn-label" style="font-family:'IBM Plex Mono',monospace;color:#a1a1aa;font-size:11px;white-space:nowrap;">Layer 0</span>
      <div style="margin-left:auto;display:flex;gap:8px;align-items:center;">
        <span id="attn-peak" style="font-family:'IBM Plex Mono',monospace;color:#fbbf24;font-size:11px;"></span>
        ${isSynthetic ? '' : '<span style="font-size:10px;color:#10b981;font-family:\'IBM Plex Mono\',monospace;border:1px solid rgba(16,185,129,0.3);padding:1px 6px;border-radius:4px;">REAL ATTN</span>'}
      </div>
    `;
    container.appendChild(ctrlRow);

    // ── Main heatmap SVG ──
    const W = (leftSide.clientWidth || 700) - 4;
    const H = 430;
    const M = { top: 32, right: 16, bottom: 36, left: 36 };
    const w = W - M.left - M.right;
    const h = H - M.top - M.bottom;

    const svg = d3.select(leftSide).append("svg").attr("width", W).attr("height", H);
    panelTitle(svg, isSynthetic ? "Activation Heatmap (Layer-by-Layer)" : "Attention Atlas — Incoming Influence per Cell");

    const g = svg.append("g").attr("transform", `translate(${M.left},${M.top})`);

    const xScale = d3.scaleBand().domain(d3.range(boardSize)).range([0, w]).padding(0.02);
    const yScale = d3.scaleBand().domain(d3.range(boardSize)).range([0, h]).padding(0.02);
    const colorTurbo = d3.scaleSequential(d3.interpolateTurbo).domain([0, 1]);
    const colorVir   = d3.scaleSequential(d3.interpolateInferno).domain([0, 1]);

    const cellData = [];
    for (let r = 0; r < boardSize; r++)
        for (let c = 0; c < boardSize; c++)
            cellData.push({ r, c, idx: r * boardSize + c, value: 0 });

    const rects = g.selectAll("rect.attn-cell").data(cellData).join("rect")
        .attr("class", "attn-cell")
        .attr("x",      d => xScale(d.c))
        .attr("y",      d => yScale(d.r))
        .attr("width",  xScale.bandwidth())
        .attr("height", yScale.bandwidth())
        .attr("rx", 2)
        .attr("fill", "#18181b");

    // Axis labels (row and column indices)
    const tickEvery = boardSize <= 8 ? 1 : boardSize <= 16 ? 4 : 8;
    g.append("g").attr("transform", `translate(0,${h})`)
        .call(d3.axisBottom(xScale).tickValues(d3.range(boardSize).filter(i => i % tickEvery === 0)).tickFormat(d => `c${d}`))
        .call(axisStyle);
    g.append("g")
        .call(d3.axisLeft(yScale).tickValues(d3.range(boardSize).filter(i => i % tickEvery === 0)).tickFormat(d => `r${d}`))
        .call(axisStyle);

    // ── Right: stats panel ──
    const statsBox = document.createElement("div");
    statsBox.style.cssText = "background:rgba(9,9,11,0.6);border:1px solid rgba(255,255,255,0.06);border-radius:10px;padding:12px;flex:1;overflow-y:auto;";
    const statsTitle = document.createElement("div");
    statsTitle.style.cssText = "font-size:12px;font-weight:600;color:#e4e4e7;margin-bottom:10px;font-family:'Inter',sans-serif;";
    statsTitle.textContent = "Hotspot Cells (Top Attention)";
    statsBox.appendChild(statsTitle);
    const statsList = document.createElement("div");
    statsBox.appendChild(statsList);
    rightSide.appendChild(statsBox);

    // ── Layer progress chart (right bottom) ──
    const progressBox = document.createElement("div");
    progressBox.style.cssText = "background:rgba(9,9,11,0.6);border:1px solid rgba(255,255,255,0.06);border-radius:10px;padding:12px;flex:0 0 120px;";
    const progTitle = document.createElement("div");
    progTitle.style.cssText = "font-size:11px;font-weight:600;color:#a1a1aa;margin-bottom:8px;font-family:'Inter',sans-serif;";
    progTitle.textContent = "Max Attention per Layer";
    progressBox.appendChild(progTitle);
    const progChartDiv = document.createElement("div");
    progressBox.appendChild(progChartDiv);
    rightSide.appendChild(progressBox);

    // Draw layer max chart
    const maxPerLayer = [];
    if (isSynthetic) {
        for (const sf of synthFrames) {
            const maxVal = Math.max(...sf.incoming);
            maxPerLayer.push({ layer: sf.layer, max: maxVal });
        }
    } else {
        for (const af of attnFrames) {
            const inc = new Array(T).fill(0);
            for (let i = 0; i < af.attn.length; i++)
                for (let j = 0; j < af.attn[i].length; j++) inc[j] += af.attn[i][j];
            maxPerLayer.push({ layer: af.layer, max: Math.max(...inc) });
        }
    }

    const pcW = 280, pcH = 80;
    const pcM = { top: 4, right: 6, bottom: 18, left: 32 };
    const pcSvg = d3.select(progChartDiv).append("svg").attr("width", pcW).attr("height", pcH);
    const pcG = pcSvg.append("g").attr("transform", `translate(${pcM.left},${pcM.top})`);
    const pcw = pcW - pcM.left - pcM.right;
    const pch = pcH - pcM.top - pcM.bottom;

    const pcX = d3.scaleLinear().domain([0, maxPerLayer.length - 1]).range([0, pcw]);
    const pcY = d3.scaleLinear().domain([0, d3.max(maxPerLayer, d => d.max) * 1.1 || 1]).range([pch, 0]);

    const pcLine = d3.line().x((_, i) => pcX(i)).y(d => pcY(d.max)).curve(d3.curveCatmullRom);
    const pcArea = d3.area().x((_, i) => pcX(i)).y0(pch).y1(d => pcY(d.max)).curve(d3.curveCatmullRom);

    const gid = "attn-prog-grad-" + Date.now();
    const defs = pcSvg.append("defs");
    const gr = defs.append("linearGradient").attr("id", gid).attr("x1","0%").attr("y1","0%").attr("x2","0%").attr("y2","100%");
    gr.append("stop").attr("offset","0%").attr("stop-color","#6366f1").attr("stop-opacity", 0.2);
    gr.append("stop").attr("offset","100%").attr("stop-color","#6366f1").attr("stop-opacity", 0.01);

    pcG.append("path").datum(maxPerLayer).attr("fill", `url(#${gid})`).attr("d", pcArea);
    pcG.append("path").datum(maxPerLayer).attr("fill", "none").attr("stroke", "#6366f1").attr("stroke-width", 1.8).attr("d", pcLine);

    pcG.append("g").attr("transform", `translate(0,${pch})`)
        .call(d3.axisBottom(pcX).ticks(4).tickFormat(i => maxPerLayer[Math.round(i)]?.layer ?? ""))
        .call(ax => { ax.select(".domain").attr("stroke","rgba(255,255,255,0.05)"); ax.selectAll("text").attr("fill","#52525b").attr("font-size","8px"); ax.selectAll("line").attr("stroke","rgba(255,255,255,0.03)"); });
    pcG.append("g")
        .call(d3.axisLeft(pcY).ticks(2).tickFormat(d3.format(".1f")))
        .call(ax => { ax.select(".domain").attr("stroke","rgba(255,255,255,0.05)"); ax.selectAll("text").attr("fill","#52525b").attr("font-size","8px"); ax.selectAll("line").attr("stroke","rgba(255,255,255,0.03)"); });

    // Layer cursor line on progress chart
    const pcCursor = pcG.append("line").attr("y1", -2).attr("y2", pch + 2).attr("stroke", "#fbbf24").attr("stroke-width", 1.5).attr("stroke-dasharray", "3,2").attr("opacity", 0);

    // ── Animate ──
    let currentIdx = 0;
    let timer = null;

    const setLayer = (k) => {
        currentIdx = Math.max(0, Math.min(k, frameCount - 1));

        let incoming;
        let layerNum;
        if (isSynthetic) {
            incoming = Array.from(synthFrames[currentIdx].incoming);
            layerNum = synthFrames[currentIdx].layer;
        } else {
            const attn = attnFrames[currentIdx].attn;
            layerNum = attnFrames[currentIdx].layer;
            incoming = new Array(T).fill(0);
            for (let i = 0; i < attn.length; i++)
                for (let j = 0; j < attn[i].length; j++) incoming[j] += attn[i][j];
        }

        const maxIn = Math.max(...incoming, 1e-8);

        rects.data(cellData.map(c => ({ ...c, value: incoming[c.idx] / maxIn })))
            .transition().duration(200)
            .attr("fill", d => d.value > 0.01 ? (isSynthetic ? colorVir(d.value) : colorTurbo(d.value)) : "#18181b");

        // Update stats list (top 12 hotspot cells)
        const sorted = incoming.map((v, i) => ({ i, v })).sort((a, b) => b.v - a.v).slice(0, 12);
        statsList.innerHTML = sorted.map((item, rank) => {
            const row = Math.floor(item.i / boardSize);
            const col = item.i % boardSize;
            const pct = (item.v / maxIn * 100).toFixed(0);
            return `<div style="display:flex;align-items:center;gap:6px;padding:3px 0;border-bottom:1px solid rgba(255,255,255,0.03);">
                <span style="font-size:10px;color:#52525b;width:18px;text-align:right;font-family:'IBM Plex Mono',monospace;">${rank + 1}</span>
                <span style="font-size:11px;color:#a1a1aa;font-family:'IBM Plex Mono',monospace;flex:1;">(${row},${col})</span>
                <div style="width:60px;height:3px;background:rgba(255,255,255,0.07);border-radius:2px;overflow:hidden;">
                    <div style="width:${pct}%;height:100%;background:#6366f1;border-radius:2px;"></div>
                </div>
                <span style="font-size:10px;color:#fbbf24;font-family:'IBM Plex Mono',monospace;width:36px;text-align:right;">${item.v.toFixed(2)}</span>
            </div>`;
        }).join("");

        const peakEl = document.getElementById("attn-peak");
        const labelEl = document.getElementById("attn-label");
        if (peakEl) peakEl.textContent = `Peak ${maxIn.toFixed(3)}`;
        if (labelEl) labelEl.textContent = `Layer ${layerNum}`;

        const slider = document.getElementById("attn-layer");
        if (slider) slider.value = String(currentIdx);

        // Move cursor on progress chart
        pcCursor.attr("x1", pcX(currentIdx)).attr("x2", pcX(currentIdx)).attr("opacity", 1);
    };

    const sliderEl = document.getElementById("attn-layer");
    if (sliderEl) sliderEl.addEventListener("input", (e) => setLayer(+e.target.value));

    const playBtn = document.getElementById("attn-play");
    if (playBtn) {
        playBtn.addEventListener("click", () => {
            if (timer) {
                clearInterval(timer); timer = null;
                playBtn.textContent = "▶ Play";
                return;
            }
            playBtn.textContent = "⏸ Pause";
            timer = setInterval(() => {
                currentIdx = (currentIdx + 1) % frameCount;
                setLayer(currentIdx);
            }, 600);
        });
    }

    setLayer(0);
}

// ══════════════════════════════════════════════════════════════════════════════
// CONCEPT PROBE
// Shows what functional concepts each neuron has specialised for
// Falls back to a derived analysis when no concept_probe data is present
// ══════════════════════════════════════════════════════════════════════════════
export function renderConceptProbe(container, data) {
    container.innerHTML = "";

    const probe   = data.concept_probe?.neurons  || [];
    const summary = data.concept_probe?.summary  || {};
    const hasProbData = probe.length > 0;

    // ── If no concept_probe, derive one from activations ──
    let items, neuronRows;
    if (!hasProbData) {
        // Derive "functional concept" from which step activates each neuron most
        const STEP_NAMES  = ["Recall", "Mechanism", "Effect", "Update"];
        const STEP_COLORS = ["#6366f1", "#8b5cf6", "#14b8a6", "#0ea5e9"];
        const N = data.topology?.nodes?.length || 0;
        if (N === 0) {
            noDataNotice(container, "No topology data available.");
            return;
        }

        // Build per-step max activations for each neuron
        const stepMaxActs = [
            new Float32Array(N),
            new Float32Array(N),
            new Float32Array(N),
            new Float32Array(N),
        ];
        for (const f of data.frames || []) {
            const s = Math.min(f.step_index, 3);
            const acts = f.activations;
            for (let i = 0; i < Math.min(N, acts.length); i++) {
                if (acts[i] > stepMaxActs[s][i]) stepMaxActs[s][i] = acts[i];
            }
        }

        // Assign each neuron to its dominant step
        const counts = [0, 0, 0, 0];
        neuronRows = [];
        for (let i = 0; i < N; i++) {
            let best = 0, bestV = stepMaxActs[0][i];
            for (let s = 1; s < 4; s++) {
                if (stepMaxActs[s][i] > bestV) { bestV = stepMaxActs[s][i]; best = s; }
            }
            const totalAct = stepMaxActs[0][i] + stepMaxActs[1][i] + stepMaxActs[2][i] + stepMaxActs[3][i];
            const purity = totalAct > 0 ? bestV / totalAct : 0;
            counts[best]++;
            const orig = data.topology.nodes[i]?.original_idx ?? i;
            neuronRows.push({ original_idx: orig, best_concept: STEP_NAMES[best], purity, maxAct: bestV, stepColor: STEP_COLORS[best] });
        }
        items = STEP_NAMES.map((n, i) => ({ concept: n, count: counts[i], color: STEP_COLORS[i] }));
    } else {
        items = Object.entries(summary).map(([k, v]) => ({ concept: k, count: v }));
        neuronRows = [...probe].sort((a, b) => b.purity - a.purity).slice(0, 40);
    }

    // ── Layout ──
    const notice = document.createElement("div");
    if (!hasProbData) {
        notice.style.cssText = "padding:8px 12px;background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.2);border-radius:8px;margin-bottom:12px;font-size:11px;color:#a5b4fc;font-family:'IBM Plex Mono',monospace;";
        notice.textContent = "ℹ Derived concept assignment: each neuron is assigned to its dominant compute phase (Recall/Mechanism/Effect/Update). Provide concept_probe data for semantic concepts.";
        container.appendChild(notice);
    }

    const topRow = document.createElement("div");
    topRow.style.cssText = "display:flex;gap:0;min-height:0;flex:1;";
    const leftDiv = document.createElement("div");
    leftDiv.style.cssText = "flex:1;min-height:340px;";
    const rightDiv = document.createElement("div");
    rightDiv.style.cssText = "flex:0 0 400px;padding:8px 12px;overflow-y:auto;max-height:440px;";
    topRow.appendChild(leftDiv);
    topRow.appendChild(rightDiv);
    container.appendChild(topRow);

    // ── Bar chart ──
    const W = leftDiv.clientWidth || 700;
    const H = 360;
    const w = W - PM.left - PM.right;
    const h = H - PM.top - PM.bottom;

    const svg = d3.select(leftDiv).append("svg").attr("width", W).attr("height", H);
    panelTitle(svg, hasProbData ? "Monosemantic Concept Probe — Neurons per Concept" : "Functional Phase Assignment — Neurons per Phase");

    const g = svg.append("g").attr("transform", `translate(${PM.left},${PM.top + 6})`);
    const adjH = h - 10;

    const xScale = d3.scaleBand().domain(items.map(d => d.concept)).range([0, w]).padding(0.28);
    const yScale = d3.scaleLinear().domain([0, d3.max(items, d => d.count) || 1]).nice().range([adjH, 0]);

    gridLines(g, yScale, w, 4);

    const COLORS = hasProbData
        ? ["#0ea5e9", "#f59e0b", "#10b981", "#f43f5e", "#8b5cf6", "#ec4899"]
        : (items.map(d => d.color));

    const colScale = hasProbData
        ? d3.scaleOrdinal().domain(items.map(d => d.concept)).range(COLORS)
        : (d => d.color);

    g.selectAll("rect.cbar").data(items).join("rect")
        .attr("class", "cbar")
        .attr("x", d => xScale(d.concept))
        .attr("y", d => yScale(d.count))
        .attr("width", xScale.bandwidth())
        .attr("height", d => adjH - yScale(d.count))
        .attr("rx", 4)
        .attr("fill", hasProbData ? d => colScale(d.concept) : d => d.color)
        .attr("opacity", 0.82)
        .append("title").text(d => `${d.concept}: ${d.count} neurons`);

    // Value labels on top of bars
    g.selectAll("text.barval").data(items).join("text")
        .attr("class", "barval")
        .attr("x", d => xScale(d.concept) + xScale.bandwidth() / 2)
        .attr("y", d => yScale(d.count) - 5)
        .attr("text-anchor", "middle")
        .attr("font-size", "11px")
        .attr("font-family", "'IBM Plex Mono', monospace")
        .attr("font-weight", "600")
        .attr("fill", "#e4e4e7")
        .text(d => d.count);

    g.append("g").attr("transform", `translate(0,${adjH})`)
        .call(d3.axisBottom(xScale)).call(axisStyle);
    g.append("g")
        .call(d3.axisLeft(yScale).ticks(5).tickFormat(d3.format("d"))).call(axisStyle);

    g.append("text").attr("x", w / 2).attr("y", adjH + 38)
        .attr("fill", "#71717a").attr("font-size", "10px")
        .attr("text-anchor", "middle").text("Concept / Phase");
    g.append("text").attr("transform", "rotate(-90)").attr("x", -adjH / 2).attr("y", -46)
        .attr("fill", "#71717a").attr("font-size", "10px")
        .attr("text-anchor", "middle").text("Neuron count");

    // ── Right: neuron table ──
    const topNeurons = hasProbData
        ? neuronRows.slice(0, 35)
        : neuronRows.sort((a, b) => b.purity - a.purity).slice(0, 35);

    const CONCEPT_COLORS_MAP = {};
    items.forEach((it, i) => { CONCEPT_COLORS_MAP[it.concept] = hasProbData ? COLORS[i % COLORS.length] : it.color; });

    rightDiv.innerHTML = `
      <div style="font-size:12px;font-weight:600;color:#e4e4e7;margin-bottom:10px;font-family:'Inter',sans-serif;">
          ${hasProbData ? "Top Neurons by Concept Purity" : "Top Neurons by Phase Purity"}
      </div>
      <div style="display:grid;grid-template-columns:30px 1fr 60px 60px;gap:0;font-size:10px;color:#52525b;font-family:'IBM Plex Mono',monospace;padding-bottom:6px;border-bottom:1px solid rgba(255,255,255,0.05);">
          <span>#</span><span>Neuron</span><span>Concept</span><span style="text-align:right;">Purity</span>
      </div>
      ${topNeurons.map((n, i) => {
          const cc = CONCEPT_COLORS_MAP[n.best_concept] || "#a1a1aa";
          const pct = (n.purity * 100).toFixed(1);
          return `<div style="display:grid;grid-template-columns:30px 1fr 60px 60px;gap:0;align-items:center;padding:3px 0;border-bottom:1px solid rgba(255,255,255,0.025);font-family:'IBM Plex Mono',monospace;font-size:10px;">
              <span style="color:#52525b">${i + 1}</span>
              <span style="color:#a1a1aa">N${n.original_idx}</span>
              <span style="color:${cc};font-weight:600;">${n.best_concept}</span>
              <span style="color:#fbbf24;text-align:right;">${pct}%</span>
          </div>`;
      }).join("")}
    `;

    // ── Bottom: aggregate stats ──
    const botDiv = document.createElement("div");
    botDiv.style.cssText = "margin-top:12px;padding:10px 12px;background:rgba(9,9,11,0.5);border:1px solid rgba(255,255,255,0.05);border-radius:8px;";
    const totalNeurons = items.reduce((s, d) => s + d.count, 0);
    const dominantConcept = items.reduce((a, b) => a.count > b.count ? a : b, items[0]);
    const avgPurity = topNeurons.length > 0 ? topNeurons.reduce((s, n) => s + n.purity, 0) / topNeurons.length : 0;
    botDiv.innerHTML = `
        <div style="display:flex;flex-wrap:wrap;gap:10px;">
            <div style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.2);border-radius:8px;padding:8px 12px;min-width:120px;">
                <div style="font-size:9px;color:#71717a;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:3px;">Total Neurons</div>
                <div style="font-size:18px;font-weight:700;color:#a5b4fc;font-family:'IBM Plex Mono',monospace;">${totalNeurons.toLocaleString()}</div>
            </div>
            <div style="background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.2);border-radius:8px;padding:8px 12px;min-width:140px;">
                <div style="font-size:9px;color:#71717a;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:3px;">Dominant ${hasProbData ? "Concept" : "Phase"}</div>
                <div style="font-size:14px;font-weight:700;color:#6ee7b7;font-family:'IBM Plex Mono',monospace;">${dominantConcept?.concept || "—"}</div>
                <div style="font-size:9px;color:#52525b;">${dominantConcept?.count} neurons</div>
            </div>
            <div style="background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.2);border-radius:8px;padding:8px 12px;min-width:130px;">
                <div style="font-size:9px;color:#71717a;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:3px;">Avg Top-35 Purity</div>
                <div style="font-size:18px;font-weight:700;color:#fbbf24;font-family:'IBM Plex Mono',monospace;">${(avgPurity * 100).toFixed(1)}%</div>
            </div>
        </div>
    `;
    container.appendChild(botDiv);
}

// ══════════════════════════════════════════════════════════════════════════════
// SCALING LAB
// BDH O(T) vs Transformer O(T²) — interactive complexity comparison
// ══════════════════════════════════════════════════════════════════════════════
export function renderScalingLab(container, data) {
    container.innerHTML = "";

    const M_neurons = data.topology?.nodes?.length || data.config?.M_neurons || 1000;
    const num_layers = data.config?.num_layers || data.frames?.[data.frames?.length - 1]?.layer + 1 || 12;
    const board_size = data.config?.board_size || 16;
    const T_max_grid = board_size * board_size;

    // ── Info cards ──
    const infoRow = document.createElement("div");
    infoRow.style.cssText = "display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px;";
    infoRow.innerHTML = `
        <div style="flex:1;min-width:160px;padding:10px 14px;background:rgba(244,63,94,0.08);border:1px solid rgba(244,63,94,0.2);border-radius:10px;">
            <div style="font-size:9px;text-transform:uppercase;letter-spacing:0.12em;color:#f43f5e;margin-bottom:4px;font-weight:700;">Transformer KV-Attention</div>
            <div style="font-size:20px;font-weight:700;color:#f43f5e;font-family:'IBM Plex Mono',monospace;">O(T²)</div>
            <div style="font-size:10px;color:#71717a;margin-top:4px;">Memory ∝ T×T per layer<br>Bottleneck at long context</div>
        </div>
        <div style="flex:1;min-width:160px;padding:10px 14px;background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.2);border-radius:10px;">
            <div style="font-size:9px;text-transform:uppercase;letter-spacing:0.12em;color:#10b981;margin-bottom:4px;font-weight:700;">BDH Hebbian State</div>
            <div style="font-size:20px;font-weight:700;color:#10b981;font-family:'IBM Plex Mono',monospace;">O(T)</div>
            <div style="font-size:10px;color:#71717a;margin-top:4px;">State = M neurons × L layers<br>Fixed memory regardless of T</div>
        </div>
        <div style="flex:1;min-width:160px;padding:10px 14px;background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.2);border-radius:10px;">
            <div style="font-size:9px;text-transform:uppercase;letter-spacing:0.12em;color:#fbbf24;margin-bottom:4px;font-weight:700;">This Model</div>
            <div style="font-size:14px;font-weight:700;color:#fbbf24;font-family:'IBM Plex Mono',monospace;">${M_neurons.toLocaleString()} × ${num_layers}</div>
            <div style="font-size:10px;color:#71717a;margin-top:4px;">${(M_neurons * num_layers).toLocaleString()} fixed state units<br>vs T²=${(T_max_grid*T_max_grid).toLocaleString()} for grid</div>
        </div>
    `;
    container.appendChild(infoRow);

    // ── Slider control ──
    const ctrlRow = document.createElement("div");
    ctrlRow.style.cssText = "display:flex;align-items:center;gap:12px;padding:10px 14px;background:rgba(9,9,11,0.6);border:1px solid rgba(255,255,255,0.06);border-radius:10px;margin-bottom:12px;flex-wrap:wrap;";
    ctrlRow.innerHTML = `
        <label style="font-size:11px;color:#a1a1aa;font-family:'IBM Plex Mono',monospace;white-space:nowrap;">Context Length T =</label>
        <input id="scale-t" type="range" min="64" max="262144" step="64" value="4096" style="flex:1;min-width:160px;cursor:pointer;">
        <span id="scale-t-label" style="font-family:'IBM Plex Mono',monospace;color:#e4e4e7;font-size:13px;font-weight:700;min-width:60px;text-align:right;">4,096</span>
        <div style="height:28px;width:1px;background:rgba(255,255,255,0.08);margin:0 4px;"></div>
        <span id="scale-kv" style="font-family:'IBM Plex Mono',monospace;color:#f43f5e;font-size:11px;white-space:nowrap;"></span>
        <span id="scale-hebb" style="font-family:'IBM Plex Mono',monospace;color:#10b981;font-size:11px;white-space:nowrap;"></span>
        <span id="scale-speedup" style="font-family:'IBM Plex Mono',monospace;color:#fbbf24;font-size:11px;white-space:nowrap;border:1px solid rgba(245,158,11,0.3);padding:2px 8px;border-radius:5px;"></span>
    `;
    container.appendChild(ctrlRow);

    // ── Main Chart ──
    const chartDiv = document.createElement("div");
    chartDiv.style.cssText = "min-height:360px;";
    container.appendChild(chartDiv);

    const W = chartDiv.clientWidth || 1000;
    const H = 390;
    const w = W - PM.left - PM.right;
    const h = H - PM.top - PM.bottom;

    const svg = d3.select(chartDiv).append("svg").attr("width", W).attr("height", H);
    panelTitle(svg, "Scaling Comparison: Transformer O(T²) vs BDH O(T)");
    const g = svg.append("g").attr("transform", `translate(${PM.left},${PM.top + 8})`);

    const ts = [64, 200, 500, 1000, 3000, 8000, 20000, 64000, 200000, 262144];
    const quadPoints = ts.map(t => ({ t, v: t * t }));
    const linPoints  = ts.map(t => ({ t, v: t }));

    const xLog = d3.scaleLog().domain([64, 262144]).range([0, w]).nice();
    const yLog = d3.scaleLog().domain([64 * 64, 262144 * 262144]).range([h, 0]);

    g.append("g").attr("transform", `translate(0,${h})`)
        .call(d3.axisBottom(xLog)
            .tickValues([64, 256, 1024, 4096, 16384, 65536, 262144])
            .tickFormat(d => d >= 1000 ? (d / 1000).toFixed(0) + "K" : d))
        .call(axisStyle);

    g.append("g")
        .call(d3.axisLeft(yLog)
            .tickValues([1e4, 1e6, 1e8, 1e10, 1e12, 1e14])
            .tickFormat(d => d3.format(".0e")(d)))
        .call(axisStyle);

    g.append("text").attr("x", w / 2).attr("y", h + 38)
        .attr("fill", "#52525b").attr("font-size", "10px").attr("text-anchor", "middle")
        .text("Context Length T (tokens)");
    g.append("text").attr("transform", "rotate(-90)").attr("x", -h / 2).attr("y", -48)
        .attr("fill", "#52525b").attr("font-size", "10px").attr("text-anchor", "middle")
        .text("Compute / Memory Units");

    // Shaded region between curves (emphasizes the gap)
    const areaGap = d3.area()
        .x(d => xLog(d.t))
        .y0(d => yLog(d.t))       // BDH linear
        .y1(d => yLog(d.t * d.t)) // Transformer quadratic
        .curve(d3.curveMonotoneX);

    g.append("path").datum(ts.map(t => ({ t })))
        .attr("fill", "rgba(244,63,94,0.06)")
        .attr("d", areaGap);

    const lineGen = d3.line().x(d => xLog(d.t)).y(d => yLog(d.v)).curve(d3.curveMonotoneX);

    // Transformer O(T²)
    g.append("path").datum(quadPoints)
        .attr("fill", "none")
        .attr("stroke", "#f43f5e")
        .attr("stroke-width", 2.8)
        .attr("stroke-linecap", "round")
        .attr("d", lineGen);

    // BDH O(T)
    g.append("path").datum(linPoints)
        .attr("fill", "none")
        .attr("stroke", "#10b981")
        .attr("stroke-width", 2.8)
        .attr("stroke-linecap", "round")
        .attr("d", lineGen);

    // BDH fixed state line
    const bdhStateY = M_neurons * num_layers;
    if (bdhStateY > 64 * 64 && bdhStateY < 262144 * 262144) {
        g.append("line")
            .attr("x1", 0).attr("x2", w)
            .attr("y1", yLog(bdhStateY)).attr("y2", yLog(bdhStateY))
            .attr("stroke", "#fbbf24").attr("stroke-width", 1.5)
            .attr("stroke-dasharray", "6,3").attr("opacity", 0.7);
        g.append("text")
            .attr("x", w - 4).attr("y", yLog(bdhStateY) - 5)
            .attr("fill", "#fbbf24").attr("font-size", "10px")
            .attr("font-family", "'IBM Plex Mono', monospace")
            .attr("text-anchor", "end")
            .text(`BDH state = ${(M_neurons * num_layers).toLocaleString()}`);
    }

    // Labels
    g.append("text").attr("x", xLog(50000)).attr("y", yLog(50000 * 50000) - 8)
        .attr("fill", "#f43f5e").attr("font-size", "11px").attr("font-family", "'IBM Plex Mono', monospace")
        .text("Transformer (T²)");
    g.append("text").attr("x", xLog(50000)).attr("y", yLog(50000) - 8)
        .attr("fill", "#10b981").attr("font-size", "11px").attr("font-family", "'IBM Plex Mono', monospace")
        .text("BDH (T)");

    // Marker line (current T)
    const markerLine = g.append("line")
        .attr("y1", 0).attr("y2", h)
        .attr("stroke", "#fbbf24").attr("stroke-width", 1.5)
        .attr("stroke-dasharray", "4,3").attr("opacity", 0.7);

    const markerDotQ = g.append("circle").attr("r", 5).attr("fill", "#f43f5e").attr("opacity", 0.9);
    const markerDotL = g.append("circle").attr("r", 5).attr("fill", "#10b981").attr("opacity", 0.9);

    const updateT = (t) => {
        const kvUnits = t * t;
        const hebbUnits = M_neurons * num_layers;
        const speedup = kvUnits / hebbUnits;

        document.getElementById("scale-t-label").textContent = t >= 1000 ? (t / 1000).toFixed(1) + "K" : t.toLocaleString();
        document.getElementById("scale-kv").textContent    = `Transformer: ${kvUnits >= 1e6 ? (kvUnits / 1e6).toFixed(1) + "M" : kvUnits.toLocaleString()}`;
        document.getElementById("scale-hebb").textContent  = `BDH: ${hebbUnits.toLocaleString()}`;
        document.getElementById("scale-speedup").textContent = `${speedup >= 1000 ? (speedup / 1000).toFixed(1) + "K" : speedup.toFixed(0)}× faster`;

        const clampedT = Math.max(64, Math.min(t, 262144));
        const xPos = xLog(clampedT);

        markerLine.attr("x1", xPos).attr("x2", xPos);

        // Clamp dots to chart range
        const yQ = Math.max(yLog.range()[1], Math.min(yLog.range()[0], yLog(clampedT * clampedT)));
        const yL = Math.max(yLog.range()[1], Math.min(yLog.range()[0], yLog(clampedT)));
        markerDotQ.attr("cx", xPos).attr("cy", yQ);
        markerDotL.attr("cx", xPos).attr("cy", yL);
    };

    const slider = document.getElementById("scale-t");
    slider.addEventListener("input", (e) => updateT(+e.target.value));
    updateT(+slider.value);

    // ── Practical comparison table ──
    const tableDiv = document.createElement("div");
    tableDiv.style.cssText = "margin-top:14px;";
    const tableData = [
        { t: 256,     label: "256 (small)" },
        { t: 1024,    label: "1K (GPT-2)" },
        { t: 4096,    label: "4K (GPT-4)" },
        { t: 32768,   label: "32K (Claude)" },
        { t: 131072,  label: "128K (long-ctx)" },
    ];
    const hebbFixed = M_neurons * num_layers;
    tableDiv.innerHTML = `
        <div style="font-size:11px;font-weight:600;color:#a1a1aa;margin-bottom:8px;font-family:'Inter',sans-serif;">Practical Comparison</div>
        <div style="display:grid;grid-template-columns:130px 1fr 1fr 80px;gap:4px;font-family:'IBM Plex Mono',monospace;font-size:10px;">
            <div style="color:#52525b;padding:3px 6px;">Context T</div>
            <div style="color:#f43f5e;padding:3px 6px;">Transformer T²</div>
            <div style="color:#10b981;padding:3px 6px;">BDH M×L</div>
            <div style="color:#fbbf24;padding:3px 6px;">Speedup</div>
            ${tableData.map(row => {
                const kv = row.t * row.t;
                const sp = (kv / hebbFixed).toFixed(0);
                return `<div style="color:#a1a1aa;padding:3px 6px;border-top:1px solid rgba(255,255,255,0.03);">${row.label}</div>
                         <div style="color:#f87171;padding:3px 6px;border-top:1px solid rgba(255,255,255,0.03);">${kv >= 1e6 ? (kv/1e6).toFixed(1)+"M" : kv.toLocaleString()}</div>
                         <div style="color:#6ee7b7;padding:3px 6px;border-top:1px solid rgba(255,255,255,0.03);">${hebbFixed.toLocaleString()}</div>
                         <div style="color:#fbbf24;padding:3px 6px;border-top:1px solid rgba(255,255,255,0.03);font-weight:600;">${+sp >= 1000 ? (+sp/1000).toFixed(1)+"K×" : sp+"×"}</div>`;
            }).join("")}
        </div>
    `;
    container.appendChild(tableDiv);
}
