// ──────────────────────────────────────────────────────────────────────────────
// panels.js — Analysis panels: Sparse Brain, Graph Topology, Memory Formation
// Professional data-centric styling — no neon, flat fills, muted palette
// ──────────────────────────────────────────────────────────────────────────────

const M = { top: 32, right: 24, bottom: 46, left: 56 };
const STEP_COLORS = ['#6366f1', '#8b5cf6', '#14b8a6', '#0ea5e9'];

// Shared chart helpers
function panelSvg(container, extraH = 0) {
    const W = container.clientWidth || 900;
    const H = container.clientHeight || 340 + extraH;
    const svg = d3.select(container).append('svg')
        .attr('width', W).attr('height', H);
    return { svg, W, H, w: W - M.left - M.right, h: H - M.top - M.bottom };
}

function axisStyle(sel) {
    sel.select('.domain').attr('stroke', 'rgba(255,255,255,0.04)');
    sel.selectAll('text').attr('fill', '#a1a1aa').attr('font-size', '11px').attr('font-family', "'IBM Plex Mono', monospace");
    sel.selectAll('line').attr('stroke', 'rgba(255,255,255,0.04)');
}

function gridH(g, scale, w, ticks = 5) {
    g.append('g').attr('class', 'grid')
        .call(d3.axisLeft(scale).ticks(ticks).tickSize(-w).tickFormat(''))
        .call(gg => { gg.select('.domain').remove(); gg.selectAll('line').attr('stroke', 'rgba(255,255,255,0.03)').attr('stroke-dasharray', '2,4'); });
}

function chartTitle(svg, W, text) {
    svg.append('text').attr('x', M.left).attr('y', 18)
        .attr('fill', '#f4f4f5').attr('font-size', '13px').attr('font-weight', '600')
        .attr('font-family', "'Inter', sans-serif").text(text);
}

// ══════════════════════════════════════════════════════════════════════════════
// SPARSE BRAIN
// ══════════════════════════════════════════════════════════════════════════════
export function renderSparseBrain(container, data) {
    container.innerHTML = '';

    const frames = data.frames;
    const byLayer = new Map();
    for (const f of frames) {
        if (!byLayer.has(f.layer)) byLayer.set(f.layer, {});
        byLayer.get(f.layer)[f.step_index] = f;
    }

    const layers = Array.from(byLayer.keys()).sort((a, b) => a - b);
    const sparsityData = layers.map(l => {
        const lf = byLayer.get(l);
        const recall = lf[0] || Object.values(lf)[0];
        const effect = lf[2] || recall;

        const act = recall.activations;
        const prev = effect.prev_activations;
        const xPct = act.filter(v => v > 0).length / act.length * 100;
        const yPct = prev.filter(v => v > 0).length / prev.length * 100;
        return { layer: l, xPct, yPct };
    });

    // Layout: two charts + stat row
    const topDiv = document.createElement('div');
    topDiv.style.cssText = 'flex:1;min-height:0;display:flex;gap:0;';
    container.appendChild(topDiv);

    const leftDiv = document.createElement('div');
    const rightDiv = document.createElement('div');
    leftDiv.style.cssText = rightDiv.style.cssText = 'flex:1;min-height:220px;';
    topDiv.appendChild(leftDiv);
    topDiv.appendChild(rightDiv);

    const botDiv = document.createElement('div');
    botDiv.style.cssText = 'flex:0 0 180px;';
    container.appendChild(botDiv);

    // LEFT: grouped bar — x% vs y% per layer
    {
        const W = leftDiv.clientWidth || 520;
        const H = leftDiv.clientHeight || 240;
        const w = W - M.left - M.right;
        const h = H - M.top - M.bottom;

        const svg = d3.select(leftDiv).append('svg').attr('width', W).attr('height', H);
        const g = svg.append('g').attr('transform', `translate(${M.left},${M.top})`);
        chartTitle(svg, W, 'Neuron Activation % per Layer');

        const x0 = d3.scaleBand().domain(layers).range([0, w]).paddingInner(0.25).paddingOuter(0.1);
        const x1 = d3.scaleBand().domain(['x', 'y']).range([0, x0.bandwidth()]).padding(0.08);
        const y = d3.scaleLinear().domain([0, Math.max(d3.max(sparsityData, d => d.xPct), 1)]).range([h, 0]).nice();

        gridH(g, y, w, 4);

        const layer_g = g.selectAll('.layer-group').data(sparsityData).join('g')
            .attr('class', 'layer-group')
            .attr('transform', d => `translate(${x0(d.layer)},0)`);

        layer_g.append('rect').attr('x', x1('x')).attr('y', d => y(d.xPct))
            .attr('width', x1.bandwidth()).attr('height', d => h - y(d.xPct))
            .attr('fill', '#6366f1').attr('rx', 2).attr('opacity', 0.7)
            .append('title').text(d => `Layer ${d.layer} x: ${d.xPct.toFixed(1)}%`);

        layer_g.append('rect').attr('x', x1('y')).attr('y', d => y(d.yPct))
            .attr('width', x1.bandwidth()).attr('height', d => h - y(d.yPct))
            .attr('fill', '#14b8a6').attr('rx', 2).attr('opacity', 0.7)
            .append('title').text(d => `Layer ${d.layer} y: ${d.yPct.toFixed(1)}%`);

        g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x0).tickFormat(d3.format('d'))).call(axisStyle);
        g.append('g').call(d3.axisLeft(y).ticks(4).tickFormat(d => d.toFixed(0) + '%')).call(axisStyle);

        // Legend
        const leg = svg.append('g').attr('transform', `translate(${M.left + w - 120},${M.top - 14})`);
        leg.append('rect').attr('width', 10).attr('height', 10).attr('rx', 2).attr('fill', '#6366f1').attr('opacity', 0.7);
        leg.append('text').attr('x', 14).attr('y', 9).attr('fill', '#a1a1aa').attr('font-size', '10px').text('x active (recall)');
        leg.append('rect').attr('y', 16).attr('width', 10).attr('height', 10).attr('rx', 2).attr('fill', '#14b8a6').attr('opacity', 0.7);
        leg.append('text').attr('x', 14).attr('y', 25).attr('fill', '#a1a1aa').attr('font-size', '10px').text('y active (hebbian)');
    }

    // RIGHT: area chart — mean activation magnitude per layer
    {
        const W = rightDiv.clientWidth || 520;
        const H = rightDiv.clientHeight || 240;
        const w = W - M.left - M.right;
        const h = H - M.top - M.bottom;

        const magData = sparsityData.map(d => {
            const lf = byLayer.get(d.layer);
            const recall = lf[0] || Object.values(lf)[0];
            const vals = recall.activations.filter(v => v > 0);
            const mean = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
            return { layer: d.layer, mean };
        });

        const svg = d3.select(rightDiv).append('svg').attr('width', W).attr('height', H);
        const g = svg.append('g').attr('transform', `translate(${M.left},${M.top})`);
        chartTitle(svg, W, 'Mean Active Magnitude per Layer');

        const x = d3.scaleLinear().domain([layers[0], layers[layers.length - 1]]).range([0, w]);
        const y = d3.scaleLinear().domain([0, d3.max(magData, d => d.mean) * 1.15 || 1]).range([h, 0]).nice();

        gridH(g, y, w, 4);

        const area = d3.area().x(d => x(d.layer)).y0(h).y1(d => y(d.mean)).curve(d3.curveCatmullRom);
        const line = d3.line().x(d => x(d.layer)).y(d => y(d.mean)).curve(d3.curveCatmullRom);

        const gid = 'mag-grad-' + Date.now();
        const defs = svg.append('defs');
        const gr = defs.append('linearGradient').attr('id', gid).attr('x1', '0%').attr('y1', '0%').attr('x2', '0%').attr('y2', '100%');
        gr.append('stop').attr('offset', '0%').attr('stop-color', '#0ea5e9').attr('stop-opacity', 0.10);
        gr.append('stop').attr('offset', '100%').attr('stop-color', '#0ea5e9').attr('stop-opacity', 0.01);

        g.append('path').datum(magData).attr('fill', `url(#${gid})`).attr('d', area);
        g.append('path').datum(magData).attr('fill', 'none').attr('stroke', '#0ea5e9').attr('stroke-width', 2.5).attr('stroke-linecap', 'round').attr('d', line);
        g.selectAll('circle').data(magData).join('circle')
            .attr('cx', d => x(d.layer)).attr('cy', d => y(d.mean)).attr('r', 3)
            .attr('fill', '#0ea5e9').attr('stroke', '#09090b').attr('stroke-width', 1.5)
            .append('title').text(d => `Layer ${d.layer}: ${d.mean.toFixed(4)}`);

        g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(layers.length).tickFormat(d3.format('d'))).call(axisStyle);
        g.append('g').call(d3.axisLeft(y).ticks(4).tickFormat(d => d.toFixed(2))).call(axisStyle);
        g.append('text').attr('x', w / 2).attr('y', h + 36).attr('fill', '#71717a').attr('font-size', '10px').attr('text-anchor', 'middle').text('Layer');
    }

    // BOTTOM: stat summary
    {
        const allX = sparsityData.map(d => d.xPct);
        const allY = sparsityData.map(d => d.yPct);
        const s = (v, d = 2) => v.toFixed(d);
        botDiv.innerHTML = `
        <div class="stat-row">
            <div class="stat-chip"><span class="label">x avg active</span><span class="val purple">${s(allX.reduce((a, b) => a + b, 0) / allX.length)}%</span></div>
            <div class="stat-chip"><span class="label">y avg active</span><span class="val green">${s(allY.reduce((a, b) => a + b, 0) / allY.length)}%</span></div>
            <div class="stat-chip"><span class="label">x max active</span><span class="val amber">${s(Math.max(...allX))}%</span></div>
            <div class="stat-chip"><span class="label">y max active</span><span class="val amber">${s(Math.max(...allY))}%</span></div>
            <div class="stat-chip"><span class="label">total neurons</span><span class="val">${sparsityData[0] ? data.topology.nodes.length : '—'}</span></div>
        </div>`;
    }
}


// ══════════════════════════════════════════════════════════════════════════════
// GRAPH TOPOLOGY
// ══════════════════════════════════════════════════════════════════════════════

// ── Label propagation community detection ──
function detectCommunities(N, edges) {
    const labels = new Int32Array(N);
    for (let i = 0; i < N; i++) labels[i] = i;

    const adj = new Array(N);
    for (let i = 0; i < N; i++) adj[i] = [];
    for (const e of edges) {
        const s = typeof e.source === 'object' ? e.source.index : +e.source;
        const t = typeof e.target === 'object' ? e.target.index : +e.target;
        adj[s].push(t);
        adj[t].push(s);
    }

    for (let iter = 0; iter < 12; iter++) {
        let changed = false;
        const order = Array.from({ length: N }, (_, i) => i);
        for (let i = order.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [order[i], order[j]] = [order[j], order[i]];
        }
        for (const i of order) {
            if (adj[i].length === 0) continue;
            const cnt = {};
            for (const j of adj[i]) cnt[labels[j]] = (cnt[labels[j]] || 0) + 1;
            let best = labels[i], bestC = 0;
            for (const [l, c] of Object.entries(cnt)) {
                if (c > bestC || (c === bestC && +l < best)) { best = +l; bestC = c; }
            }
            if (best !== labels[i]) { labels[i] = best; changed = true; }
        }
        if (!changed) break;
    }

    const map = new Map();
    let nxt = 0;
    const ids = new Int32Array(N);
    for (let i = 0; i < N; i++) {
        if (!map.has(labels[i])) map.set(labels[i], nxt++);
        ids[i] = map.get(labels[i]);
    }
    return { ids, count: nxt };
}

const COMMUNITY_PALETTE = [
    '#6366f1', '#0ea5e9', '#14b8a6', '#f59e0b', '#ef4444',
    '#8b5cf6', '#10b981', '#f97316', '#ec4899', '#06b6d4',
    '#a855f7', '#84cc16', '#fb923c', '#f43e5e', '#22d3ee',
];

export function renderGraphTopology(container, topology) {
    container.innerHTML = '';

    // ── Section 1: Interactive force-directed community graph ──
    const graphDiv = document.createElement('div');
    graphDiv.style.cssText = 'width:100%;height:400px;position:relative;margin-bottom:16px;border-radius:12px;overflow:hidden;border:1px solid rgba(255,255,255,0.05);background:#09090b;';
    container.appendChild(graphDiv);

    // Graph title overlay
    const gTitle = document.createElement('div');
    gTitle.style.cssText = 'position:absolute;top:12px;left:16px;z-index:10;display:flex;align-items:center;gap:10px;';
    gTitle.innerHTML = `<span style="font-size:13px;font-weight:600;color:#f4f4f5;font-family:Inter,sans-serif;">Interactive Topology — Community Structure</span>`;
    graphDiv.appendChild(gTitle);

    // Compute communities
    const N = topology.nodes.length;
    const comm = detectCommunities(N, topology.edges);

    // Compute degree
    const degreeArr = new Array(N).fill(0);
    topology.edges.forEach(e => {
        const s = typeof e.source === 'object' ? e.source.index : +e.source;
        const t = typeof e.target === 'object' ? e.target.index : +e.target;
        if (s < N) degreeArr[s]++;
        if (t < N) degreeArr[t]++;
    });
    const maxDeg = Math.max(...degreeArr, 1);

    // Clone data for independent D3 simulation
    const miniNodes = topology.nodes.map((n, i) => ({
        ...n, index: i, degree: degreeArr[i], community: comm.ids[i],
    }));
    const miniEdges = topology.edges.map(e => ({
        source: typeof e.source === 'object' ? e.source.index : +e.source,
        target: typeof e.target === 'object' ? e.target.index : +e.target,
        weight: e.weight || 1,
    }));

    const GW = graphDiv.clientWidth || 900;
    const GH = 400;

    const gSvg = d3.select(graphDiv).append('svg')
        .attr('width', GW).attr('height', GH)
        .style('background', '#09090b');

    // Defs for subtle glow
    const gDefs = gSvg.append('defs');
    const gGlow = gDefs.append('filter').attr('id', 'topo-glow').attr('x', '-50%').attr('y', '-50%').attr('width', '200%').attr('height', '200%');
    gGlow.append('feGaussianBlur').attr('stdDeviation', '2').attr('result', 'b');
    const gM = gGlow.append('feMerge');
    gM.append('feMergeNode').attr('in', 'b');
    gM.append('feMergeNode').attr('in', 'SourceGraphic');

    const gZoom = d3.zoom().scaleExtent([0.2, 6]).on('zoom', ev => gG.attr('transform', ev.transform));
    gSvg.call(gZoom);

    const gG = gSvg.append('g');

    // Force simulation
    const sim = d3.forceSimulation(miniNodes)
        .force('link', d3.forceLink(miniEdges).id(d => d.index).distance(30).strength(0.06))
        .force('charge', d3.forceManyBody().strength(-60))
        .force('center', d3.forceCenter(GW / 2, GH / 2))
        .force('x', d3.forceX(GW / 2).strength(0.05))
        .force('y', d3.forceY(GH / 2).strength(0.05))
        .force('collide', d3.forceCollide(d => 3 + (d.degree / maxDeg) * 8));

    // Links
    const gLink = gG.append('g').selectAll('line').data(miniEdges).join('line')
        .attr('stroke', 'rgba(160,190,255,0.06)')
        .attr('stroke-width', 0.4);

    // Nodes
    const gNode = gG.append('g').selectAll('g').data(miniNodes).join('g')
        .attr('cursor', 'pointer')
        .call(d3.drag()
            .on('start', (ev) => { if (!ev.active) sim.alphaTarget(0.3).restart(); ev.subject.fx = ev.subject.x; ev.subject.fy = ev.subject.y; })
            .on('drag', (ev) => { ev.subject.fx = ev.x; ev.subject.fy = ev.y; })
            .on('end', (ev) => { if (!ev.active) sim.alphaTarget(0); ev.subject.fx = null; ev.subject.fy = null; })
        );

    // Outer glow ring for high-degree
    gNode.append('circle')
        .attr('r', d => 3 + (d.degree / maxDeg) * 10)
        .attr('fill', d => {
            const c = COMMUNITY_PALETTE[d.community % COMMUNITY_PALETTE.length];
            return d.degree > maxDeg * 0.5 ? c : 'none';
        })
        .attr('opacity', 0.12)
        .style('filter', 'url(#topo-glow)');

    // Core circle
    gNode.append('circle')
        .attr('r', d => 1.8 + (d.degree / maxDeg) * 5)
        .attr('fill', d => COMMUNITY_PALETTE[d.community % COMMUNITY_PALETTE.length])
        .attr('opacity', 0.8)
        .attr('stroke', d => d.degree > maxDeg * 0.4 ? 'rgba(255,255,255,0.3)' : 'none')
        .attr('stroke-width', 0.5);

    // Tooltip
    gNode.append('title')
        .text(d => `Neuron ${d.original_idx ?? d.id}\nDegree: ${d.degree}\nCommunity: ${d.community}`);

    sim.on('tick', () => {
        gLink.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
             .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
        gNode.attr('transform', d => `translate(${d.x},${d.y})`);
    });

    // Community legend (floating)
    const topCommunities = Array.from({ length: Math.min(comm.count, 8) }, (_, i) => i);
    const legDiv = document.createElement('div');
    legDiv.style.cssText = 'position:absolute;bottom:12px;right:16px;z-index:10;display:flex;flex-wrap:wrap;gap:6px;background:rgba(9,9,11,0.7);backdrop-filter:blur(8px);border:1px solid rgba(255,255,255,0.05);border-radius:8px;padding:6px 10px;';
    legDiv.innerHTML = topCommunities.map(i =>
        `<span style="display:flex;align-items:center;gap:3px;font-size:10px;color:#a1a1aa;">` +
        `<span style="width:7px;height:7px;border-radius:50%;background:${COMMUNITY_PALETTE[i]}"></span>C${i}</span>`
    ).join('') + `<span style="font-size:10px;color:#71717a;margin-left:4px;">${comm.count} communities</span>`;
    graphDiv.appendChild(legDiv);

    // ── Section 2: Analytics row (degree histogram + hub table) ──
    const topDiv = document.createElement('div');
    topDiv.style.cssText = 'flex:1;min-height:0;display:flex;gap:0;';
    container.appendChild(topDiv);

    const leftDiv = document.createElement('div');
    const rightDiv = document.createElement('div');
    leftDiv.style.cssText = 'flex:1;min-height:240px;';
    rightDiv.style.cssText = 'flex:0 0 320px;overflow-y:auto;padding:20px 16px;';
    topDiv.appendChild(leftDiv);
    topDiv.appendChild(rightDiv);

    const botDiv = document.createElement('div');
    botDiv.style.cssText = 'flex:0 0 140px;';
    container.appendChild(botDiv);

    // Reuse degreeArr already computed above
    const degArr = degreeArr.map((deg, i) => [i, deg]);
    const degVals = degreeArr;

    // LEFT: degree distribution histogram
    {
        const W = leftDiv.clientWidth || 520;
        const H = leftDiv.clientHeight || 260;
        const w = W - M.left - M.right;
        const h = H - M.top - M.bottom;

        const svg = d3.select(leftDiv).append('svg').attr('width', W).attr('height', H);
        const g = svg.append('g').attr('transform', `translate(${M.left},${M.top})`);
        chartTitle(svg, W, 'Degree Distribution');

        const x = d3.scaleLinear().domain([0, d3.max(degVals) || 1]).range([0, w]);
        const bins = d3.bin().domain(x.domain()).thresholds(20)(degVals);
        const yMax = d3.max(bins, d => d.length);
        const y = d3.scaleLinear().domain([0, yMax]).range([h, 0]).nice();

        gridH(g, y, w, 4);

        const colorScale = d3.scaleSequential(d3.interpolateRgb('#6366f1', '#14b8a6'))
            .domain([0, d3.max(degVals) || 1]);

        g.selectAll('rect').data(bins).join('rect')
            .attr('x', d => x(d.x0) + 1)
            .attr('y', d => y(d.length))
            .attr('width', d => Math.max(0, x(d.x1) - x(d.x0) - 1))
            .attr('height', d => h - y(d.length))
            .attr('fill', d => colorScale((d.x0 + d.x1) / 2))
            .attr('rx', 2).attr('opacity', 0.7)
            .append('title').text(d => `Degree ${d.x0}–${d.x1}: ${d.length} neurons`);

        g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(8)).call(axisStyle);
        g.append('g').call(d3.axisLeft(y).ticks(4).tickFormat(d3.format('d'))).call(axisStyle);
        g.append('text').attr('x', w / 2).attr('y', h + 36).attr('fill', '#71717a').attr('font-size', '10px').attr('text-anchor', 'middle').text('Degree (connections)');
        g.append('text').attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', -42).attr('fill', '#71717a').attr('font-size', '10px').attr('text-anchor', 'middle').text('Neuron count');
    }

    // RIGHT: Hub neuron table
    {
        const sorted = degArr.slice().sort((a, b) => b[1] - a[1]).slice(0, 15);
        const maxDeg = sorted[0]?.[1] || 1;

        rightDiv.innerHTML = `<div class="hub-table-title">Top Hub Neurons</div>
        <table class="hub-table">
        <thead><tr><th>#</th><th>Neuron</th><th>Degree</th><th></th></tr></thead>
        <tbody>
        ${sorted.map(([idx, deg], i) => {
            const n = topology.nodes[idx];
            const label = n ? (n.original_idx ?? n.id ?? idx) : idx;
            const pct = (deg / maxDeg * 100).toFixed(0);
            const barColor = i < 3 ? '#0ea5e9' : '#6366f1';
            return `<tr>
                <td class="rank">${i + 1}</td>
                <td>${label}</td>
                <td class="deg-val">${deg}</td>
                <td><div class="deg-bar" style="width:${pct}%;background:${barColor}"></div></td>
            </tr>`;
        }).join('')}
        </tbody></table>`;
    }

    // BOTTOM: topology stats
    {
        const mean = degVals.reduce((a, b) => a + b, 0) / (degVals.length || 1);
        const max = Math.max(...degVals);
        const isolated = degVals.filter(d => d === 0).length;
        botDiv.innerHTML = `
        <div class="stat-row">
            <div class="stat-chip"><span class="label">nodes</span><span class="val">${topology.nodes.length}</span></div>
            <div class="stat-chip"><span class="label">edges</span><span class="val purple">${topology.edges.length}</span></div>
            <div class="stat-chip"><span class="label">avg degree</span><span class="val amber">${mean.toFixed(2)}</span></div>
            <div class="stat-chip"><span class="label">max degree</span><span class="val amber">${max}</span></div>
            <div class="stat-chip"><span class="label">communities</span><span class="val green">${comm.count}</span></div>
            <div class="stat-chip"><span class="label">isolated</span><span class="val">${isolated}</span></div>
        </div>`;
    }
}



// ══════════════════════════════════════════════════════════════════════════════
// MEMORY FORMATION (Hebbian Dynamics)
// ══════════════════════════════════════════════════════════════════════════════
export function renderMemoryFormation(container, data) {
    container.innerHTML = '';

    const topDiv = document.createElement('div');
    topDiv.style.cssText = 'flex:1;min-height:0;display:flex;gap:0;';
    container.appendChild(topDiv);

    const leftDiv = document.createElement('div');
    const rightDiv = document.createElement('div');
    leftDiv.style.cssText = 'flex:1;min-height:240px;';
    rightDiv.style.cssText = 'flex:0 0 340px;min-height:240px;';
    topDiv.appendChild(leftDiv);
    topDiv.appendChild(rightDiv);

    const botDiv = document.createElement('div');
    botDiv.style.cssText = 'flex:0 0 140px;';
    container.appendChild(botDiv);

    // Compute per-layer Hebbian stats
    const frames = data.frames;
    const byLayer = new Map();
    for (const f of frames) {
        if (!byLayer.has(f.layer)) byLayer.set(f.layer, {});
        byLayer.get(f.layer)[f.step_index] = f;
    }
    const layers = Array.from(byLayer.keys()).sort((a, b) => a - b);

    let cumStrength = 0;
    const hebbData = layers.map(l => {
        const lf = byLayer.get(l);
        const effect = lf[2] || lf[1] || lf[0] || Object.values(lf)[0];
        const recall = lf[0] || Object.values(lf)[0];

        const y = effect.prev_activations;
        const x = recall.activations;

        const yActive = y.filter(v => v > 0);
        const hebbScore = yActive.length ? yActive.reduce((a, b) => a + b, 0) / yActive.length : 0;
        const coact = x.filter((v, i) => v > 0 && y[i] > 0).length / x.length;
        const yMean = y.reduce((a, b) => a + Math.abs(b), 0) / y.length;

        cumStrength += yMean;

        return { layer: l, hebbScore, coact, yMean, cumStrength };
    });

    // LEFT: cumulative synapse strength line
    {
        const W = leftDiv.clientWidth || 520;
        const H = leftDiv.clientHeight || 260;
        const w = W - M.left - M.right;
        const h = H - M.top - M.bottom;

        const svg = d3.select(leftDiv).append('svg').attr('width', W).attr('height', H);
        const g = svg.append('g').attr('transform', `translate(${M.left},${M.top})`);
        chartTitle(svg, W, 'Cumulative Synapse Strength  Σ|y|');

        const x = d3.scaleLinear().domain([layers[0], layers[layers.length - 1]]).range([0, w]);
        const y = d3.scaleLinear().domain([0, hebbData[hebbData.length - 1].cumStrength * 1.1 || 1]).range([h, 0]).nice();

        gridH(g, y, w, 5);

        const gid = 'hebb-grad-' + Date.now();
        const defs = svg.append('defs');
        const gr = defs.append('linearGradient').attr('id', gid).attr('x1', '0%').attr('y1', '0%').attr('x2', '0%').attr('y2', '100%');
        gr.append('stop').attr('offset', '0%').attr('stop-color', '#8b5cf6').attr('stop-opacity', 0.10);
        gr.append('stop').attr('offset', '100%').attr('stop-color', '#8b5cf6').attr('stop-opacity', 0.01);

        const area = d3.area().x(d => x(d.layer)).y0(h).y1(d => y(d.cumStrength)).curve(d3.curveCatmullRom);
        const line = d3.line().x(d => x(d.layer)).y(d => y(d.cumStrength)).curve(d3.curveCatmullRom);

        g.append('path').datum(hebbData).attr('fill', `url(#${gid})`).attr('d', area);
        g.append('path').datum(hebbData).attr('fill', 'none').attr('stroke', '#8b5cf6').attr('stroke-width', 2.5).attr('stroke-linecap', 'round').attr('d', line);

        g.selectAll('circle').data(hebbData).join('circle')
            .attr('cx', d => x(d.layer)).attr('cy', d => y(d.cumStrength)).attr('r', 3.5)
            .attr('fill', '#8b5cf6').attr('stroke', '#09090b').attr('stroke-width', 1.5)
            .append('title').text(d => `Layer ${d.layer}: Σ=${d.cumStrength.toFixed(5)}`);

        g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(layers.length).tickFormat(d3.format('d'))).call(axisStyle);
        g.append('g').call(d3.axisLeft(y).ticks(5).tickFormat(d => d.toExponential(1))).call(axisStyle);
        g.append('text').attr('x', w / 2).attr('y', h + 36).attr('fill', '#71717a').attr('font-size', '10px').attr('text-anchor', 'middle').text('Layer');
    }

    // RIGHT: per-layer co-activation bars
    {
        const W = rightDiv.clientWidth || 340;
        const H = rightDiv.clientHeight || 260;
        const w = W - M.left - M.right;
        const h = H - M.top - M.bottom;

        const svg = d3.select(rightDiv).append('svg').attr('width', W).attr('height', H);
        const g = svg.append('g').attr('transform', `translate(${M.left},${M.top})`);
        chartTitle(svg, W, 'Co-activation Rate per Layer');

        const x = d3.scaleBand().domain(layers).range([0, w]).padding(0.2);
        const y = d3.scaleLinear().domain([0, d3.max(hebbData, d => d.coact) * 1.2 || 0.01]).range([h, 0]).nice();
        const cs = d3.scaleSequential(d3.interpolateRgb('#18181b', '#14b8a6')).domain([0, d3.max(hebbData, d => d.coact)]);

        gridH(g, y, w, 4);

        g.selectAll('rect').data(hebbData).join('rect')
            .attr('x', d => x(d.layer))
            .attr('y', d => y(d.coact))
            .attr('width', x.bandwidth())
            .attr('height', d => h - y(d.coact))
            .attr('fill', d => cs(d.coact))
            .attr('rx', 3)
            .append('title').text(d => `Layer ${d.layer}: ${(d.coact * 100).toFixed(2)}% co-active`);

        g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).tickFormat(d3.format('d'))).call(axisStyle);
        g.append('g').call(d3.axisLeft(y).ticks(4).tickFormat(d => (d * 100).toFixed(1) + '%')).call(axisStyle);
        g.append('text').attr('x', w / 2).attr('y', h + 36).attr('fill', '#71717a').attr('font-size', '10px').attr('text-anchor', 'middle').text('Layer');
    }

    // BOTTOM: stat chips
    {
        const finalCum = hebbData[hebbData.length - 1]?.cumStrength ?? 0;
        const avgHebb = hebbData.reduce((a, d) => a + d.hebbScore, 0) / hebbData.length;
        const avgCoact = hebbData.reduce((a, d) => a + d.coact, 0) / hebbData.length;
        const maxCoact = Math.max(...hebbData.map(d => d.coact));
        botDiv.innerHTML = `
        <div class="stat-row">
            <div class="stat-chip"><span class="label">final Σ|y|</span><span class="val purple">${finalCum.toExponential(3)}</span></div>
            <div class="stat-chip"><span class="label">avg hebbian score</span><span class="val purple">${avgHebb.toFixed(4)}</span></div>
            <div class="stat-chip"><span class="label">avg co-activation</span><span class="val green">${(avgCoact * 100).toFixed(2)}%</span></div>
            <div class="stat-chip"><span class="label">peak co-activation</span><span class="val amber">${(maxCoact * 100).toFixed(2)}%</span></div>
        </div>`;
    }
}
