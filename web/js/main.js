import { initGraph, updateGraph, cleanupGraph, setOnSelect, setIntensity } from './graph.js';
import { initBoard, updateBoard } from './board.js';
import { initChart, drawChart, updateCursor } from './chart.js';
import { renderSparseBrain, renderGraphTopology, renderMemoryFormation } from './panels.js';
import { initHebbianPanel, setHebbianActive } from './hebbian.js';
import { init3DScene, update3DFrame, set3DActive, cleanup3D, toggleAutoRotate } from './viz3d.js';
import { initInferenceViz } from './inference_viz.js';
import { renderAttentionAtlas, renderConceptProbe, renderScalingLab } from './advanced_panels.js';

// ── State ─────────────────────────────────────────────────────────────────────
let state = {
    data: null,
    currentFrame: 0,
    isPlaying: false,
    playbackInterval: null,
    playbackSpeed: 600,
    selection: null,
    activeTab: 'graph',
    hebbianRendered: false,
    threeRendered: false,
    inferenceRendered: false,
    liveAnalysisRendered: false,
    panelsRendered: {
        sparse: false,
        topology: false,
        memory: false,
        attention: false,
        concept: false,
        scaling: false,
    },
    histSvg: null,
    histX: null,
    histBars: null,
    totalNeurons: 0,
    // Track max peak activation seen (for stable bar scaling)
    maxPeakEver: 3.0,
};

const STEP_CLASSES = ['step-recall', 'step-mechanism', 'step-effect', 'step-update'];
const STEP_LABELS  = ['RECALL', 'MECHANISM', 'EFFECT', 'UPDATE'];
const STEP_PILL_CLASSES = ['recall', 'mechanism', 'effect', 'update'];

// ── DOM refs ──────────────────────────────────────────────────────────────────
const els = {
    layerDisplay:    document.getElementById('layer-display'),
    totalLayers:     document.getElementById('total-layers'),
    scrubber:        document.getElementById('scrubber'),
    btnPrev:         document.getElementById('btn-prev'),
    btnNext:         document.getElementById('btn-next'),
    btnPlay:         document.getElementById('btn-play'),
    iconPlay:        document.getElementById('icon-play'),
    iconPause:       document.getElementById('icon-pause'),
    playText:        document.getElementById('play-text'),
    neuronCount:     document.getElementById('neuron-count'),
    edgeCount:       document.getElementById('edge-count'),
    infoPanel:       document.getElementById('info-panel'),
    stepBadge:       document.getElementById('step-badge'),
    stepBadgePill:   document.getElementById('step-badge-pill'),
    stepCounter:     document.getElementById('step-counter'),
    stepTitle:       document.getElementById('step-title'),
    stepDescription: document.getElementById('step-description'),
    chartPanel:      document.getElementById('chart-panel'),
    chartInner:      document.getElementById('chart-inner'),
    chartLabel:      document.getElementById('chart-label'),
    chartClose:      document.getElementById('chart-close'),
    // Live telemetry
    teleActive:      document.getElementById('tele-active'),
    teleActiveBar:   document.getElementById('tele-active-bar'),
    teleSparsity:    document.getElementById('tele-sparsity'),
    teleSparsityBar: document.getElementById('tele-sparsity-bar'),
    telePeak:        document.getElementById('tele-peak'),
    telePeakBar:     document.getElementById('tele-peak-bar'),
    teleMean:        document.getElementById('tele-mean'),
    teleMeanBar:     document.getElementById('tele-mean-bar'),
    teleLayers:      document.getElementById('tele-layers'),
    actHistogram:    document.getElementById('act-histogram'),
    gridFrameLabel:  document.getElementById('grid-frame-label'),
    graphNodeCount:  document.getElementById('graph-node-count'),
    graphEdgeCount:  document.getElementById('graph-edge-count'),
    pathQualityPct:  document.getElementById('path-quality-pct'),
    pathQualityFill: document.getElementById('path-quality-fill'),
    // Step pips
    layerStepRow:    document.getElementById('layer-step-row'),
    // Speed control
    speedSelect:     document.getElementById('speed-select'),
};

// ── Activation Histogram (D3 mini chart in sidebar) ───────────────────────────
function initActHistogram() {
    const container = els.actHistogram;
    if (!container) return;
    container.innerHTML = '';

    const W = container.clientWidth || 200;
    const H = 60;
    const M = { top: 4, right: 4, bottom: 14, left: 24 };
    const w = W - M.left - M.right;
    const h = H - M.top - M.bottom;

    const svg = d3.select(container).append('svg')
        .attr('width', W).attr('height', H)
        .style('overflow', 'visible');

    const g = svg.append('g').attr('transform', `translate(${M.left},${M.top})`);

    // X axis: activation bins 0..3
    const NUM_BINS = 20;
    const xScale = d3.scaleLinear().domain([0, 3]).range([0, w]);

    // Y axis: count (dynamic)
    const yScale = d3.scaleLinear().domain([0, 100]).range([h, 0]);

    // Draw axis
    g.append('g').attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xScale).ticks(3).tickFormat(d => d.toFixed(1)))
        .call(ax => {
            ax.select('.domain').attr('stroke', 'rgba(255,255,255,0.06)');
            ax.selectAll('text').attr('fill', '#52525b').attr('font-size', '8px');
            ax.selectAll('line').attr('stroke', 'rgba(255,255,255,0.04)');
        });

    g.append('g')
        .call(d3.axisLeft(yScale).ticks(2).tickFormat(d3.format('d')))
        .call(ax => {
            ax.select('.domain').attr('stroke', 'rgba(255,255,255,0.06)');
            ax.selectAll('text').attr('fill', '#52525b').attr('font-size', '8px');
            ax.selectAll('line').attr('stroke', 'rgba(255,255,255,0.04)');
        });

    const binWidth = w / NUM_BINS;
    const bars = g.selectAll('rect.hbar')
        .data(d3.range(NUM_BINS))
        .join('rect')
        .attr('class', 'hbar')
        .attr('x', i => i * binWidth + 0.5)
        .attr('y', h)
        .attr('width', Math.max(1, binWidth - 1))
        .attr('height', 0)
        .attr('rx', 1)
        .attr('fill', '#6366f1')
        .attr('opacity', 0.7);

    state.histSvg = svg;
    state.histG = g;
    state.histX = xScale;
    state.histY = yScale;
    state.histBars = bars;
    state.histBinWidth = binWidth;
    state.histNumBins = NUM_BINS;
    state.histH = h;
}

function updateActHistogram(activations) {
    if (!state.histBars || !activations) return;

    const NUM_BINS = state.histNumBins;
    const maxVal   = 3.0;
    const counts   = new Array(NUM_BINS).fill(0);

    for (let i = 0; i < activations.length; i++) {
        const v = activations[i];
        if (v <= 0) continue;
        const bin = Math.min(Math.floor((v / maxVal) * NUM_BINS), NUM_BINS - 1);
        counts[bin]++;
    }

    const maxCount = Math.max(...counts, 1);
    state.histY.domain([0, maxCount]);

    state.histBars
        .data(counts)
        .transition()
        .duration(200)
        .attr('y',      c => state.histY(c))
        .attr('height', c => state.histH - state.histY(c))
        .attr('fill', (c, i) => {
            // Color by activation range
            const t = i / NUM_BINS;
            if (t < 0.25) return '#6366f1';
            if (t < 0.5)  return '#0ea5e9';
            if (t < 0.75) return '#f59e0b';
            return '#fefce8';
        });
}

// ── Live Telemetry Update ─────────────────────────────────────────────────────
function updateLiveTelemetry(frame) {
    if (!frame || !state.data) return;
    const act = frame.activations;
    const N   = act.length;
    state.totalNeurons = N;

    const THRESH = 0.01;
    let activeCount = 0;
    let sum = 0;
    let maxV = 0;
    for (let i = 0; i < N; i++) {
        const v = act[i];
        if (v > THRESH) { activeCount++; sum += v; }
        if (v > maxV) maxV = v;
    }
    const sparsity = ((1 - activeCount / N) * 100);
    const meanNonZero = activeCount > 0 ? sum / activeCount : 0;

    // Update max peak seen (so bars don't rescale erratically)
    if (maxV > state.maxPeakEver) state.maxPeakEver = maxV;

    // Update DOM
    if (els.teleActive)    els.teleActive.textContent    = activeCount.toLocaleString();
    if (els.teleSparsity)  els.teleSparsity.textContent  = sparsity.toFixed(1) + '%';
    if (els.telePeak)      els.telePeak.textContent      = maxV.toFixed(3);
    if (els.teleMean)      els.teleMean.textContent      = meanNonZero.toFixed(3);

    const activeRatio = N > 0 ? (activeCount / N) : 0;
    if (els.teleActiveBar)  els.teleActiveBar.style.width    = (activeRatio * 100).toFixed(1) + '%';
    if (els.teleSparsityBar) els.teleSparsityBar.style.width = sparsity.toFixed(1) + '%';
    if (els.telePeakBar)    els.telePeakBar.style.width      = Math.min((maxV / state.maxPeakEver) * 100, 100).toFixed(1) + '%';
    if (els.teleMeanBar)    els.teleMeanBar.style.width      = Math.min((meanNonZero / state.maxPeakEver) * 100, 100).toFixed(1) + '%';

    // Update grid frame label
    if (els.gridFrameLabel) els.gridFrameLabel.textContent = frame.layer;

    // Path quality
    if (frame.board_prediction) {
        const pathCells = frame.board_prediction.filter(v => v === 4).length;
        const maxPathCells = 30; // approx max path length in 16x16 grid
        const quality = Math.min((pathCells / maxPathCells) * 100, 100);
        if (els.pathQualityPct)  els.pathQualityPct.textContent  = pathCells + ' cells';
        if (els.pathQualityFill) els.pathQualityFill.style.width = quality.toFixed(1) + '%';
    }

    // Update activation histogram
    updateActHistogram(act);

    // Update live analysis tab if active
    if (state.activeTab === 'liveanalysis') {
        renderLiveAnalysisFrame(frame);
    }
}

// ── Live Analysis Panel ────────────────────────────────────────────────────────
function renderLiveAnalysisFrame(frame) {
    const container = document.getElementById('liveanalysis-body');
    if (!container || !frame || !state.data) return;
    container.innerHTML = '';

    const act  = frame.activations;
    const prev = frame.prev_activations || act;
    const N    = act.length;

    const THRESH = 0.01;
    const activeIdx = [];
    for (let i = 0; i < N; i++) if (act[i] > THRESH) activeIdx.push(i);
    activeIdx.sort((a, b) => act[b] - act[a]);

    // ── Section 1: Current Step Info ──
    const stepNames = ['Recall Phase', 'Mechanism (Gate)', 'Effect (Hebbian Update)', 'Residual Update'];
    const stepDesc = [
        'x = ReLU(v* · Dx) — Sparse feature recall from learned weight matrix D. Only top-k neurons activate.',
        'Gx = E ⊗ Dx — Encoder E gates the recall signal. Sparsity is enforced by the binary mask.',
        'y(l+1) = α·y(l) + η·(y(l)·W^T)⊙x(l) — Hebbian trace update. y strengthens paths that fire together.',
        'Residual stream update — Signal added back to the token representation for next layer.',
    ];
    const stepIdx = frame.step_index || 0;

    const stepDiv = document.createElement('div');
    stepDiv.className = 'bg-zinc-950 rounded-xl border border-zinc-800 p-4';
    stepDiv.innerHTML = `
        <div class="flex items-center gap-3 mb-3">
            <span class="step-badge-pill ${STEP_PILL_CLASSES[stepIdx]}">${STEP_LABELS[stepIdx]}</span>
            <span class="text-sm font-medium text-zinc-200">Layer ${frame.layer}</span>
            <span class="text-xs text-zinc-500 ml-auto font-mono">Frame ${state.currentFrame + 1} / ${state.data.frames.length}</span>
        </div>
        <div class="font-mono text-xs text-indigo-300 bg-indigo-500/5 border border-indigo-500/15 rounded-lg p-3 mb-2">${stepNames[stepIdx] || ''}</div>
        <p class="text-xs text-zinc-400 leading-relaxed">${stepDesc[stepIdx] || frame.description || ''}</p>
    `;
    container.appendChild(stepDiv);

    // ── Section 2: Activation Stats ──
    let maxV = 0, sumV = 0;
    for (let i = 0; i < N; i++) { if (act[i] > maxV) maxV = act[i]; if (act[i] > THRESH) sumV += act[i]; }
    const meanNZ = activeIdx.length > 0 ? (sumV / activeIdx.length) : 0;
    const sparsity = ((1 - activeIdx.length / N) * 100);

    // Compute delta stats
    let deltaPos = 0, deltaNeg = 0;
    for (let i = 0; i < N; i++) {
        const d = act[i] - (prev[i] || 0);
        if (d > 0.001) deltaPos++;
        else if (d < -0.001) deltaNeg++;
    }

    const statsDiv = document.createElement('div');
    statsDiv.className = 'grid grid-cols-3 gap-3';
    const statsItems = [
        { label: 'Active Neurons', value: activeIdx.length.toLocaleString(), color: 'text-indigo-300', sub: `of ${N}` },
        { label: 'Sparsity', value: sparsity.toFixed(1) + '%', color: 'text-teal-300', sub: 'inactive' },
        { label: 'Peak Activation', value: maxV.toFixed(4), color: 'text-amber-300', sub: 'max val' },
        { label: 'Mean (active)', value: meanNZ.toFixed(4), color: 'text-sky-300', sub: 'avg nonzero' },
        { label: 'Rising Neurons', value: deltaPos.toLocaleString(), color: 'text-emerald-300', sub: 'Δ > 0' },
        { label: 'Falling Neurons', value: deltaNeg.toLocaleString(), color: 'text-rose-300', sub: 'Δ < 0' },
    ];
    statsDiv.innerHTML = statsItems.map(s => `
        <div class="bg-zinc-950 rounded-lg p-3 border border-zinc-800">
            <div class="text-[9px] uppercase font-bold tracking-wider text-zinc-600 mb-1">${s.label}</div>
            <div class="text-base font-bold font-mono ${s.color}">${s.value}</div>
            <div class="text-[9px] text-zinc-600 mt-0.5">${s.sub}</div>
        </div>
    `).join('');
    container.appendChild(statsDiv);

    // ── Section 3: Top Active Neurons Table ──
    const topN = activeIdx.slice(0, 20);
    if (topN.length > 0) {
        const tableDiv = document.createElement('div');
        tableDiv.className = 'bg-zinc-950 rounded-xl border border-zinc-800 p-4';
        tableDiv.innerHTML = `
            <div class="flex items-center justify-between mb-3">
                <span class="text-xs font-bold uppercase tracking-[0.12em] text-zinc-500">Top Active Neurons</span>
                <span class="text-[10px] text-zinc-600 font-mono">${topN.length} shown of ${activeIdx.length}</span>
            </div>
            <div class="space-y-1">
                ${topN.map((idx, rank) => {
                    const v = act[idx];
                    const dv = v - (prev[idx] || 0);
                    const pct = Math.min((v / maxV) * 100, 100);
                    const delta = dv > 0.001 ? `<span class="text-emerald-400 text-[9px]">+${dv.toFixed(3)}</span>` :
                                  dv < -0.001 ? `<span class="text-rose-400 text-[9px]">${dv.toFixed(3)}</span>` :
                                  `<span class="text-zinc-600 text-[9px]">~</span>`;
                    const orig = state.data.topology.nodes[idx]?.original_idx ?? idx;
                    return `<div class="flex items-center gap-2 py-0.5">
                        <span class="text-[9px] text-zinc-600 w-4 text-right font-mono">${rank + 1}</span>
                        <span class="text-[10px] text-zinc-400 font-mono w-10">N${orig}</span>
                        <div class="flex-1 h-2 bg-zinc-800 rounded-full overflow-hidden">
                            <div class="h-full rounded-full bg-indigo-500" style="width:${pct.toFixed(1)}%;opacity:0.8;"></div>
                        </div>
                        <span class="text-[10px] text-zinc-300 font-mono w-14 text-right">${v.toFixed(4)}</span>
                        <span class="w-12 text-right">${delta}</span>
                    </div>`;
                }).join('')}
            </div>
        `;
        container.appendChild(tableDiv);
    }

    // ── Section 4: Activation Distribution Histogram (larger) ──
    const histDiv = document.createElement('div');
    histDiv.className = 'bg-zinc-950 rounded-xl border border-zinc-800 p-4';
    histDiv.innerHTML = `<div class="text-xs font-bold uppercase tracking-[0.12em] text-zinc-500 mb-3">Activation Value Distribution</div>`;
    const histContainer = document.createElement('div');
    histContainer.style.height = '120px';
    histDiv.appendChild(histContainer);
    container.appendChild(histDiv);

    // Draw larger histogram with D3
    const W2 = histContainer.clientWidth || 600;
    const H2 = 120;
    const M2 = { top: 8, right: 12, bottom: 24, left: 36 };
    const w2 = W2 - M2.left - M2.right;
    const h2 = H2 - M2.top - M2.bottom;

    const maxBinVal = Math.max(maxV, 3.0);
    const NUM_BINS2 = 30;
    const bins2 = new Array(NUM_BINS2).fill(0);
    for (let i = 0; i < N; i++) {
        const v = act[i];
        if (v <= 0.001) continue;
        const b = Math.min(Math.floor((v / maxBinVal) * NUM_BINS2), NUM_BINS2 - 1);
        bins2[b]++;
    }
    const maxCnt2 = Math.max(...bins2, 1);

    const svg2 = d3.select(histContainer).append('svg').attr('width', W2).attr('height', H2);
    const g2 = svg2.append('g').attr('transform', `translate(${M2.left},${M2.top})`);

    const xS2 = d3.scaleLinear().domain([0, maxBinVal]).range([0, w2]);
    const yS2 = d3.scaleLinear().domain([0, maxCnt2]).range([h2, 0]).nice();

    // Grid lines
    g2.append('g').call(d3.axisLeft(yS2).ticks(3).tickSize(-w2).tickFormat(''))
        .call(ag => { ag.select('.domain').remove(); ag.selectAll('line').attr('stroke', 'rgba(255,255,255,0.03)').attr('stroke-dasharray', '2,4'); });

    const bw2 = w2 / NUM_BINS2;
    const colorScale2 = d3.scaleSequential()
        .domain([0, NUM_BINS2 - 1])
        .interpolator(d3.interpolateRgbBasis(['#6366f1', '#0ea5e9', '#f59e0b', '#fefce8']));

    g2.selectAll('rect').data(bins2).join('rect')
        .attr('x', (_, i) => i * bw2 + 0.5)
        .attr('y', c => yS2(c))
        .attr('width', Math.max(1, bw2 - 1))
        .attr('height', c => h2 - yS2(c))
        .attr('rx', 1.5)
        .attr('fill', (_, i) => colorScale2(i))
        .attr('opacity', 0.75);

    g2.append('g').attr('transform', `translate(0,${h2})`)
        .call(d3.axisBottom(xS2).ticks(5).tickFormat(d => d.toFixed(1)))
        .call(ax => {
            ax.select('.domain').attr('stroke', 'rgba(255,255,255,0.06)');
            ax.selectAll('text').attr('fill', '#71717a').attr('font-size', '9px');
            ax.selectAll('line').attr('stroke', 'rgba(255,255,255,0.04)');
        });
    g2.append('g')
        .call(d3.axisLeft(yS2).ticks(3).tickFormat(d3.format('d')))
        .call(ax => {
            ax.select('.domain').attr('stroke', 'rgba(255,255,255,0.06)');
            ax.selectAll('text').attr('fill', '#71717a').attr('font-size', '9px');
            ax.selectAll('line').attr('stroke', 'rgba(255,255,255,0.04)');
        });

    g2.append('text').attr('x', w2 / 2).attr('y', h2 + 20)
        .attr('fill', '#52525b').attr('font-size', '9px').attr('text-anchor', 'middle').text('Activation value');
}

// ── Load data ─────────────────────────────────────────────────────────────────
async function loadData() {
    try {
        const response = await fetch('data/viz_data.json?v=' + Date.now());
        if (!response.ok) throw new Error('Failed to load data');
        state.data = await response.json();

        initUI();
        setOnSelect(handleSelection);
        initGraph(document.getElementById('graph-container'), state.data.topology);
        initBoard(document.getElementById('board-container'), state.data.config.board_size, state.data.input_board);
        initChart(els.chartInner);
        initActHistogram();
        setFrame(0);

        // Update graph header counts
        if (els.graphNodeCount) els.graphNodeCount.textContent = state.data.topology.nodes.length.toLocaleString();
        if (els.graphEdgeCount) els.graphEdgeCount.textContent = state.data.topology.edges.length.toLocaleString();
        if (els.teleLayers)     els.teleLayers.textContent     = state.data.frames.length;

        // Pre-render panels in background
        setTimeout(() => renderStaticPanels(), 400);

        console.log('Viz loaded. Frames:', state.data.frames.length, '| Nodes:', state.data.topology.nodes.length);
    } catch (err) {
        console.error(err);
        if (els.stepDescription) els.stepDescription.innerHTML =
            'Run <code style="background:#1e293b;padding:2px 6px;border-radius:4px;color:#60a5fa;">python utils/export_viz_data.py</code> then refresh.';
    }
}

function renderStaticPanels() {
    if (!state.data) return;

    if (!state.panelsRendered.sparse) {
        renderSparseBrain(document.getElementById('sparse-body'), state.data);
        state.panelsRendered.sparse = true;
    }
    if (!state.panelsRendered.topology) {
        renderGraphTopology(document.getElementById('topology-body'), state.data.topology);
        state.panelsRendered.topology = true;
    }
    if (!state.panelsRendered.memory) {
        renderMemoryFormation(document.getElementById('memory-body'), state.data);
        state.panelsRendered.memory = true;
    }
    if (!state.panelsRendered.attention) {
        renderAttentionAtlas(document.getElementById('attention-body'), state.data);
        state.panelsRendered.attention = true;
    }
    if (!state.panelsRendered.concept) {
        renderConceptProbe(document.getElementById('concept-body'), state.data);
        state.panelsRendered.concept = true;
    }
    if (!state.panelsRendered.scaling) {
        renderScalingLab(document.getElementById('scaling-body'), state.data);
        state.panelsRendered.scaling = true;
    }

    if (!state.hebbianRendered) {
        const hebbianContainer = document.getElementById('hebbian-body');
        if (hebbianContainer) {
            initHebbianPanel(hebbianContainer);
            state.hebbianRendered = true;
        }
    }

    if (!state.threeRendered) {
        const threeContainer = document.getElementById('three-container');
        if (threeContainer && state.activeTab === 'three') {
            init3DScene(threeContainer, state.data.topology);
            state.threeRendered = true;
            if (state.data.frames.length) {
                update3DFrame(state.data.frames[state.currentFrame]);
            }
            const rotBtn = document.getElementById('three-rotate-toggle');
            if (rotBtn) {
                rotBtn.addEventListener('click', () => {
                    const nowOn = toggleAutoRotate();
                    rotBtn.textContent = nowOn ? 'Auto-Rotate: ON' : 'Auto-Rotate: OFF';
                });
            }
        }
    }

    if (!state.inferenceRendered && state.activeTab === 'inference') {
        const inferenceContainer = document.getElementById('inference-body');
        if (inferenceContainer) {
            initInferenceViz(inferenceContainer);
            state.inferenceRendered = true;
        }
    }

    // Live analysis: render whenever active
    if (state.activeTab === 'liveanalysis' && state.data && state.data.frames.length) {
        renderLiveAnalysisFrame(state.data.frames[state.currentFrame]);
        state.liveAnalysisRendered = true;
    }
}

// ── Tab system ────────────────────────────────────────────────────────────────
function initTabs() {
    const btns   = document.querySelectorAll('.tab-btn');
    const panels = document.querySelectorAll('.tab-panel');

    btns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.tab;
            if (tab === state.activeTab) return;
            state.activeTab = tab;

            btns.forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
            panels.forEach(p => p.classList.toggle('active', p.id === `panel-${tab}`));

            if (state.data) renderStaticPanels();

            set3DActive(tab === 'three');
            setHebbianActive(tab === 'hebbian');

            if (tab === 'three' && state.threeRendered && state.data) {
                update3DFrame(state.data.frames[state.currentFrame]);
            }
        });
    });
}

// ── UI init ───────────────────────────────────────────────────────────────────
function initUI() {
    const maxFrame = state.data.frames.length - 1;
    els.totalLayers.textContent = state.data.frames.length;
    els.scrubber.max = maxFrame;
    els.scrubber.value = 0;
    els.neuronCount.textContent = state.data.topology.nodes.length;
    els.edgeCount.textContent   = state.data.topology.edges.length;

    els.scrubber.addEventListener('input', e => { setFrame(parseInt(e.target.value)); pause(); });
    els.btnPrev.addEventListener('click', () => { setFrame(state.currentFrame - 1); pause(); });
    els.btnNext.addEventListener('click', () => { setFrame(state.currentFrame + 1); pause(); });
    els.btnPlay.addEventListener('click', togglePlay);

    // Speed control
    if (els.speedSelect) {
        els.speedSelect.addEventListener('change', e => {
            state.playbackSpeed = parseInt(e.target.value);
            if (state.isPlaying) { pause(); play(); }
        });
    }

    document.addEventListener('keydown', e => {
        if (e.key === 'ArrowRight') { setFrame(state.currentFrame + 1); pause(); }
        else if (e.key === 'ArrowLeft')  { setFrame(state.currentFrame - 1); pause(); }
        else if (e.key === ' ') { e.preventDefault(); togglePlay(); }
        else if (e.key === 'Escape') closeChart();
    });

    if (els.chartClose) els.chartClose.addEventListener('click', closeChart);
    initTabs();
}

// ── Selection handler ─────────────────────────────────────────────────────────
function handleSelection(sel) {
    state.selection = sel;
    if (!sel) { closeChart(); return; }

    const frames = state.data.frames;
    let series;

    if (sel.type === 'neuron') {
        series = frames.map(f => ({
            layer: f.layer,
            value: f.activations[sel.index] ?? 0,
            stepName: f.step_name || STEP_LABELS[f.step_index] || '',
            stepIndex: f.step_index,
        }));
    } else if (sel.type === 'edge') {
        const w = sel.weight;
        series = frames.map(f => ({
            layer: f.layer,
            value: f.prev_activations[sel.srcIdx] * w * f.activations[sel.tgtIdx],
            stepName: f.step_name || STEP_LABELS[f.step_index] || '',
            stepIndex: f.step_index,
        }));
    }

    els.chartPanel.classList.add('visible');
    els.chartPanel.classList.remove('hidden');
    els.chartLabel.textContent = sel.label;
    const currentLayer = state.data.frames[state.currentFrame].layer;
    drawChart(series, sel.label, currentLayer);
}

function closeChart() {
    state.selection = null;
    if (els.chartPanel) {
        els.chartPanel.classList.remove('visible');
        els.chartPanel.classList.add('hidden');
    }
}

// ── Frame control ─────────────────────────────────────────────────────────────
function setFrame(frameIdx) {
    const maxFrame = state.data.frames.length - 1;
    if (frameIdx < 0) frameIdx = 0;
    if (frameIdx > maxFrame) frameIdx = maxFrame;
    state.currentFrame = frameIdx;

    const frame = state.data.frames[frameIdx];
    const stepIdx = frame.step_index;

    els.layerDisplay.textContent = frame.layer;
    els.scrubber.value = frameIdx;

    updateStepPips(stepIdx);
    updateInfoPanel(frame, stepIdx);
    updateLiveTelemetry(frame);
    render();

    if (state.selection && els.chartPanel.classList.contains('visible')) {
        updateCursor(frame.layer);
    }
}

function updateStepPips(activeStep) {
    document.querySelectorAll('.layer-step-pip').forEach((pip, i) => {
        pip.classList.remove('active', 'done');
        if (i === activeStep)     pip.classList.add('active');
        else if (i < activeStep) pip.classList.add('done');
    });
    // Legacy pips
    document.querySelectorAll('.step-pip').forEach((pip, i) => {
        pip.classList.remove('active', 'done');
        if (i === activeStep)     pip.classList.add('active');
        else if (i < activeStep) pip.classList.add('done');
    });
    document.querySelectorAll('.step-connector').forEach((conn, i) => {
        conn.classList.remove('done');
        if (i < activeStep) conn.classList.add('done');
    });
}

function updateInfoPanel(frame, stepIdx) {
    // Update step badge pill
    if (els.stepBadgePill) {
        els.stepBadgePill.textContent = STEP_LABELS[stepIdx] || '';
        els.stepBadgePill.className = `step-badge-pill ${STEP_PILL_CLASSES[stepIdx] || 'recall'}`;
    }
    // Legacy badge
    if (els.stepBadge) els.stepBadge.textContent = STEP_LABELS[stepIdx] || '';

    if (els.infoPanel) {
        STEP_CLASSES.forEach(c => els.infoPanel.classList.remove(c));
        els.infoPanel.classList.add(STEP_CLASSES[stepIdx] || STEP_CLASSES[0]);
    }
    if (els.stepCounter)     els.stepCounter.textContent     = `Step ${stepIdx + 1}/4`;
    if (els.stepTitle)       els.stepTitle.textContent       = frame.step_name || '';
    if (els.stepDescription) els.stepDescription.textContent = frame.description || '';
}

function render() {
    if (!state.data) return;
    const frame = state.data.frames[state.currentFrame];
    updateGraph(frame);
    updateBoard(frame, state.data.config.board_size);
    if (state.activeTab === 'three' && state.threeRendered) {
        update3DFrame(frame);
    }
}

// ── Playback ──────────────────────────────────────────────────────────────────
function togglePlay() { state.isPlaying ? pause() : play(); }

function play() {
    state.isPlaying = true;
    els.iconPlay.classList.add('hidden');
    els.iconPause.classList.remove('hidden');
    if (els.playText) els.playText.textContent = 'RUNNING';
    if (state.currentFrame >= state.data.frames.length - 1) setFrame(0);
    state.playbackInterval = setInterval(() => {
        if (state.currentFrame < state.data.frames.length - 1) setFrame(state.currentFrame + 1);
        else pause();
    }, state.playbackSpeed);
}

function pause() {
    state.isPlaying = false;
    els.iconPlay.classList.remove('hidden');
    els.iconPause.classList.add('hidden');
    if (els.playText) els.playText.textContent = 'RUN ENGINE';
    if (state.playbackInterval) { clearInterval(state.playbackInterval); state.playbackInterval = null; }
}

loadData();
