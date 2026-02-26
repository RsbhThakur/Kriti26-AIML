// ──────────────────────────────────────────────────────────────────────────────
// hebbian.js — Hebbian Learning Action Tab orchestrator
// "Neurons that fire together, wire together" — 30-board animated demo
//
// Layout:  ┌────────── 3D Scene ──────────┬─ Sidebar ─┐
//          │  Three.js Hebbian synapse     │ Board +   │
//          │  reinforcement visualizer     │ Stats     │
//          ├──────────────────────────────────────────────┤
//          │   Board thumbnail strip (30)  │ Timeline   │
//          └──────────────────────────────────────────────┘
// ──────────────────────────────────────────────────────────────────────────────

import { initHebbian3D, setHebbianFrame, setHebb3DActive, toggleHebb3DRotate, cleanupHebb3D } from './hebbian3d.js';

// ── State ────────────────────────────────────────────────────────────────────
let state = {
    data: null,
    frameIdx: 0,
    playing: false,
    timer: null,
    sceneReady: false,
    container: null,
};

// ── Board rendering ─────────────────────────────────────────────────────────
function cellMarkup(v) {
    if (v === 1) return '<div class="hb-cell hb-wall"></div>';
    if (v === 2) return '<div class="hb-cell hb-start">S</div>';
    if (v === 3) return '<div class="hb-cell hb-end">E</div>';
    if (v === 4) return '<div class="hb-cell hb-path"></div>';
    return '<div class="hb-cell hb-empty"></div>';
}

function renderBoard(el, flat, size) {
    if (!el || !flat) return;
    el.innerHTML = '';
    el.style.display = 'grid';
    el.style.gridTemplateColumns = `repeat(${size}, 1fr)`;
    el.style.gap = '1px';
    for (let i = 0; i < flat.length; i++) el.innerHTML += cellMarkup(flat[i]);
}

// ── Playback ─────────────────────────────────────────────────────────────────
function stopPlay() {
    state.playing = false;
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
    const btn = document.getElementById('hb-play');
    if (btn) btn.textContent = '▶  Play';
}

function startPlay() {
    if (!state.data) return;
    state.playing = true;
    const btn = document.getElementById('hb-play');
    if (btn) btn.textContent = '⏸  Pause';

    state.timer = setInterval(() => {
        if (state.frameIdx >= state.data.frames.length - 1) {
            stopPlay();
            return;
        }
        goFrame(state.frameIdx + 1);
    }, 800);
}

function togglePlay() { state.playing ? stopPlay() : startPlay(); }

// ── D3 mini timeline chart ──────────────────────────────────────────────────
let tlSVG = null;
let tlCursor = null;

function renderTimeline(containerId) {
    const el = document.getElementById(containerId);
    if (!el || !state.data) return;
    el.innerHTML = '';

    const frames = state.data.frames;
    const W = el.clientWidth || 700;
    const H = 72;
    const M = { top: 8, right: 14, bottom: 22, left: 40 };
    const w = W - M.left - M.right;
    const h = H - M.top - M.bottom;

    tlSVG = d3.select(el).append('svg').attr('width', W).attr('height', H);
    const g = tlSVG.append('g').attr('transform', `translate(${M.left},${M.top})`);

    const x = d3.scaleLinear().domain([0, frames.length - 1]).range([0, w]);
    const maxRaw = d3.max(frames, f => f.h_raw_max) || 1;
    const y = d3.scaleLinear().domain([0, maxRaw * 1.1]).range([h, 0]).nice();

    // Area
    const area = d3.area()
        .x((_, i) => x(i))
        .y0(h)
        .y1((d) => y(d.h_raw_max))
        .curve(d3.curveCatmullRom);

    const gid = 'hb-tl-grad';
    const defs = tlSVG.append('defs');
    const gr = defs.append('linearGradient').attr('id', gid).attr('x1','0%').attr('y1','0%').attr('x2','0%').attr('y2','100%');
    gr.append('stop').attr('offset','0%').attr('stop-color','#fbbf24').attr('stop-opacity', 0.25);
    gr.append('stop').attr('offset','100%').attr('stop-color','#fbbf24').attr('stop-opacity', 0.02);

    g.append('path').datum(frames).attr('fill', `url(#${gid})`).attr('d', area);

    const line = d3.line().x((_, i) => x(i)).y(d => y(d.h_raw_max)).curve(d3.curveCatmullRom);
    g.append('path').datum(frames).attr('fill','none').attr('stroke','#fbbf24').attr('stroke-width', 1.8).attr('d', line);

    // Dots for path found / not found
    g.selectAll('circle').data(frames).join('circle')
        .attr('cx', (_, i) => x(i))
        .attr('cy', d => y(d.h_raw_max))
        .attr('r', 2.5)
        .attr('fill', d => d.path_found ? '#10b981' : '#ef4444')
        .attr('stroke', '#09090b')
        .attr('stroke-width', 1);

    // Axis
    g.append('g').attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(x).ticks(Math.min(frames.length, 15)).tickFormat(i => `B${i + 1}`))
        .call(s => { s.select('.domain').attr('stroke','rgba(255,255,255,0.05)'); s.selectAll('text').attr('fill','#71717a').attr('font-size','8px'); s.selectAll('line').attr('stroke','rgba(255,255,255,0.03)'); });

    g.append('g')
        .call(d3.axisLeft(y).ticks(3).tickFormat(d3.format('.0f')))
        .call(s => { s.select('.domain').attr('stroke','rgba(255,255,255,0.05)'); s.selectAll('text').attr('fill','#71717a').attr('font-size','8px'); s.selectAll('line').attr('stroke','rgba(255,255,255,0.03)'); });

    // Cursor (vertical line)
    tlCursor = g.append('line')
        .attr('x1', x(0)).attr('x2', x(0))
        .attr('y1', -2).attr('y2', h + 2)
        .attr('stroke', '#f4f4f5').attr('stroke-width', 1.5).attr('stroke-dasharray', '3,3');

    // Label
    g.append('text').attr('x', w).attr('y', -1)
        .attr('fill','#71717a').attr('font-size','8px').attr('text-anchor','end').text('Cumulative Hebbian Max');
}

function updateTimelineCursor() {
    if (!tlCursor || !state.data) return;
    const frames = state.data.frames;
    const el = document.getElementById('hb-timeline');
    if (!el) return;
    const W = el.clientWidth || 700;
    const M = { left: 40, right: 14 };
    const w = W - M.left - M.right;
    const x = d3.scaleLinear().domain([0, frames.length - 1]).range([0, w]);
    const xp = x(state.frameIdx);
    tlCursor.attr('x1', xp).attr('x2', xp);
}

// ── Board thumbnail strip ───────────────────────────────────────────────────
function renderBoardStrip() {
    const strip = document.getElementById('hb-board-strip');
    if (!strip || !state.data) return;
    strip.innerHTML = '';

    const size = state.data.config.board_size;
    state.data.frames.forEach((frame, idx) => {
        const thumb = document.createElement('div');
        thumb.className = `hb-thumb ${idx === state.frameIdx ? 'active' : ''}`;
        thumb.title = `Board ${idx + 1} — ${frame.path_found ? '✓ Path found' : '✗ No path'}`;
        thumb.dataset.idx = idx;

        // Mini board (tiny)
        const grid = document.createElement('div');
        grid.className = 'hb-thumb-grid';
        grid.style.gridTemplateColumns = `repeat(${size}, 1fr)`;

        for (let i = 0; i < frame.board.length; i++) {
            const c = document.createElement('div');
            const v = frame.board[i];
            c.className = v === 1 ? 'tw' : v === 2 ? 'ts' : v === 3 ? 'te' : v === 4 ? 'tp' : 'tn';
            grid.appendChild(c);
        }

        const label = document.createElement('div');
        label.className = 'hb-thumb-label';
        label.innerHTML = `<span>#${idx + 1}</span><span class="${frame.path_found ? 'text-emerald-400' : 'text-red-400'}">${frame.accuracy_pct}%</span>`;

        thumb.appendChild(grid);
        thumb.appendChild(label);
        thumb.addEventListener('click', () => { stopPlay(); goFrame(idx); });
        strip.appendChild(thumb);
    });
}

function highlightThumb() {
    const thumbs = document.querySelectorAll('.hb-thumb');
    thumbs.forEach((t, i) => t.classList.toggle('active', i === state.frameIdx));
    // Scroll active thumb into view
    const active = document.querySelector('.hb-thumb.active');
    if (active) active.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
}

// ── Frame navigation ────────────────────────────────────────────────────────
function goFrame(idx) {
    if (!state.data) return;
    state.frameIdx = Math.max(0, Math.min(idx, state.data.frames.length - 1));
    const frame = state.data.frames[state.frameIdx];

    // Update 3D scene
    if (state.sceneReady) setHebbianFrame(frame);

    // Update sidebar stats
    const counterEl = document.getElementById('hb-counter');
    if (counterEl) counterEl.textContent = `${state.frameIdx + 1} / ${state.data.frames.length}`;

    const accEl = document.getElementById('hb-acc');
    if (accEl) accEl.textContent = `${frame.accuracy_pct}%`;

    const foundEl = document.getElementById('hb-found');
    if (foundEl) {
        foundEl.textContent = frame.path_found ? 'Yes' : 'No';
        foundEl.className = frame.path_found ? 'text-emerald-400 font-semibold' : 'text-red-400 font-semibold';
    }

    const hMaxEl = document.getElementById('hb-hmax');
    if (hMaxEl) hMaxEl.textContent = frame.h_raw_max.toFixed(1);

    // Strong edges count
    const strongEdges = frame.hebbian_weights.filter(w => w > 0.3).length;
    const strongEl = document.getElementById('hb-strong');
    if (strongEl) strongEl.textContent = strongEdges;

    // Peak synapse
    const peakSyn = Math.max(...frame.hebbian_weights).toFixed(2);
    const peakEl = document.getElementById('hb-peak');
    if (peakEl) peakEl.textContent = peakSyn;

    // Update boards
    const size = state.data.config.board_size;
    renderBoard(document.getElementById('hb-board-in'), frame.board, size);
    renderBoard(document.getElementById('hb-board-out'), frame.predicted_board, size);

    // Update scrubber
    const scrub = document.getElementById('hb-scrubber');
    if (scrub) scrub.value = state.frameIdx;

    // Update timeline cursor
    updateTimelineCursor();

    // Highlight thumb
    highlightThumb();
}

// ── Data generation ─────────────────────────────────────────────────────────
async function generateData() {
    const btn = document.getElementById('hebbian-generate');
    const sel = document.getElementById('hebbian-neurons');
    const neurons = parseInt(sel?.value || '1000', 10);
    if (btn) { btn.disabled = true; btn.textContent = 'Generating...'; }

    try {
        const resp = await fetch('/api/hebbian-run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ neurons, n_boards: 30 }),
        });
        const j = await resp.json();
        if (!resp.ok || j.status !== 'success') throw new Error(j.message || 'Generation failed');
        await loadData();
    } catch (e) {
        alert('Hebbian generation failed: ' + e.message);
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = 'Generate'; }
    }
}

// ── Data loading ────────────────────────────────────────────────────────────
async function loadData() {
    const resp = await fetch(`data/hebbian_data.json?v=${Date.now()}`);
    if (!resp.ok) throw new Error('hebbian_data.json not found');
    state.data = await resp.json();
    state.frameIdx = 0;

    // Update scrubber range
    const scrub = document.getElementById('hb-scrubber');
    if (scrub) scrub.max = state.data.frames.length - 1;

    // Config info
    const cfgEl = document.getElementById('hb-cfg');
    if (cfgEl) {
        const c = state.data.config;
        cfgEl.innerHTML = `${c.M_neurons} neurons &middot; ${c.n_edges} edges &middot; ${c.num_layers} layers`;
    }

    // Init 3D scene if not already done
    if (!state.sceneReady) {
        const container3d = document.getElementById('hb-3d-container');
        if (container3d) {
            initHebbian3D(container3d, state.data.topology);
            state.sceneReady = true;
        }
    }

    renderBoardStrip();
    renderTimeline('hb-timeline');
    goFrame(0);
}

// ── Scaffold HTML ───────────────────────────────────────────────────────────
function buildScaffold(container) {
    container.innerHTML = `
    <div class="hb-layout">
        <!-- Row 1: 3D Scene + Sidebar -->
        <div class="hb-row-main">
            <div class="hb-scene-wrap">
                <div id="hb-3d-container" class="hb-3d"></div>
                <!-- Overlays on 3D scene -->
                <div class="hb-scene-overlay top-left">
                    <span class="hb-olay-title">Hebbian Synapse Reinforcement</span>
                    <span class="hb-olay-sub">Watch synapses glow as boards are processed</span>
                </div>
                <div class="hb-scene-overlay bottom-left">
                    <div class="hb-legend-row"><span class="hb-dot" style="background:#18181b;border:1px solid #3f3f46"></span> Dormant</div>
                    <div class="hb-legend-row"><span class="hb-dot" style="background:#3730a3"></span> Awakening</div>
                    <div class="hb-legend-row"><span class="hb-dot" style="background:#0ea5e9"></span> Active</div>
                    <div class="hb-legend-row"><span class="hb-dot" style="background:#fbbf24"></span> Reinforced</div>
                    <div class="hb-legend-row"><span class="hb-dot" style="background:#fefce8;box-shadow:0 0 6px #fef3c7"></span> Dominant</div>
                </div>
                <div class="hb-scene-overlay top-right">
                    <button id="hb-rotate-toggle" class="hb-btn-sm">Auto-Rotate: ON</button>
                </div>
            </div>
            <div class="hb-sidebar">
                <div class="hb-stat-card">
                    <div class="hb-stat-header">Board Progress</div>
                    <div id="hb-counter" class="hb-stat-big">0 / 0</div>
                    <div id="hb-cfg" class="hb-stat-cfg"></div>
                </div>
                <div class="hb-stat-card">
                    <div class="hb-stat-header">Current Board</div>
                    <div class="hb-stat-grid">
                        <span>Path found</span><span id="hb-found" class="font-semibold">-</span>
                        <span>Accuracy</span><span id="hb-acc" class="font-semibold">-</span>
                        <span>Hebbian max</span><span id="hb-hmax" class="font-semibold">-</span>
                        <span>Strong edges</span><span id="hb-strong" class="font-semibold">-</span>
                        <span>Peak synapse</span><span id="hb-peak" class="font-semibold">-</span>
                    </div>
                </div>
                <div class="hb-stat-card">
                    <div class="hb-stat-header">Input Board</div>
                    <div id="hb-board-in" class="hb-board-preview"></div>
                </div>
                <div class="hb-stat-card">
                    <div class="hb-stat-header">Model Prediction</div>
                    <div id="hb-board-out" class="hb-board-preview"></div>
                </div>
                <div class="hb-stat-card hb-concept-box">
                    <div class="hb-stat-header" style="color:#fbbf24">The Hebbian Rule</div>
                    <p class="hb-concept-text">When neuron <em>i</em> and neuron <em>j</em> fire at the same time, their
                    connection is <strong>strengthened</strong>.</p>
                    <div class="hb-formula-pretty">
                        <span class="hf-delta">Δ</span><span class="hf-var">w</span><sub class="hf-sub">ij</sub>
                        <span class="hf-op">=</span>
                        <span class="hf-var">η</span>
                        <span class="hf-op">·</span>
                        <span class="hf-var">x</span><sub class="hf-sub">i</sub>
                        <span class="hf-op">·</span>
                        <span class="hf-var">x</span><sub class="hf-sub">j</sub>
                    </div>
                    <p class="hb-concept-text">After 30 boards, the dominant synapses reveal the <strong>memory pathways</strong>
                    the network has learned.</p>
                </div>
            </div>
        </div>
        <!-- Row 2: Controls + Timeline -->
        <div class="hb-row-controls">
            <div class="hb-playback">
                <button id="hb-prev" class="hb-btn">◀</button>
                <button id="hb-play" class="hb-btn hb-btn-play">▶  Play</button>
                <button id="hb-next" class="hb-btn">▶</button>
                <input id="hb-scrubber" type="range" min="0" max="0" value="0" class="hb-range" />
            </div>
            <div id="hb-timeline" class="hb-timeline"></div>
        </div>
        <!-- Row 3: Board thumbnail strip -->
        <div class="hb-row-strip">
            <div class="hb-strip-label">30-Board Progression — click to jump</div>
            <div id="hb-board-strip" class="hb-strip"></div>
        </div>
    </div>`;

    // Wire events
    document.getElementById('hb-prev')?.addEventListener('click', () => { stopPlay(); goFrame(state.frameIdx - 1); });
    document.getElementById('hb-next')?.addEventListener('click', () => { stopPlay(); goFrame(state.frameIdx + 1); });
    document.getElementById('hb-play')?.addEventListener('click', togglePlay);
    document.getElementById('hb-scrubber')?.addEventListener('input', e => { stopPlay(); goFrame(+e.target.value); });
    document.getElementById('hb-rotate-toggle')?.addEventListener('click', function() {
        const on = toggleHebb3DRotate();
        this.textContent = on ? 'Auto-Rotate: ON' : 'Auto-Rotate: OFF';
    });
    document.getElementById('hebbian-generate')?.addEventListener('click', generateData);
}

// ═══════════════════════════════════════════════════════════════════════════════
// PUBLIC: entry point
// ═══════════════════════════════════════════════════════════════════════════════
export function initHebbianPanel(container) {
    state.container = container;
    buildScaffold(container);

    loadData().catch(() => {
        const n = document.createElement('div');
        n.className = 'hb-notice';
        n.textContent = 'No hebbian_data.json yet. Click "Generate" above to compute 30-board Hebbian traces.';
        container.prepend(n);
    });
}

// Called from main.js when tab switches
export function setHebbianActive(active) {
    if (state.sceneReady) setHebb3DActive(active);
}
