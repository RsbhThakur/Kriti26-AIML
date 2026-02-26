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
    panelsRendered: {
        sparse: false,
        topology: false,
        memory: false,
        attention: false,
        concept: false,
        scaling: false,
    },
};

const STEP_CLASSES = ['step-recall', 'step-mechanism', 'step-effect', 'step-update'];
const STEP_LABELS = ['RECALL', 'MECHANISM', 'EFFECT', 'UPDATE'];

// ── DOM refs ──────────────────────────────────────────────────────────────────
const els = {
    layerDisplay: document.getElementById('layer-display'),
    totalLayers: document.getElementById('total-layers'),
    scrubber: document.getElementById('scrubber'),
    btnPrev: document.getElementById('btn-prev'),
    btnNext: document.getElementById('btn-next'),
    btnPlay: document.getElementById('btn-play'),
    iconPlay: document.getElementById('icon-play'),
    iconPause: document.getElementById('icon-pause'),
    neuronCount: document.getElementById('neuron-count'),
    edgeCount: document.getElementById('edge-count'),
    infoPanel: document.getElementById('info-panel'),
    stepBadge: document.getElementById('step-badge'),
    stepCounter: document.getElementById('step-counter'),
    stepTitle: document.getElementById('step-title'),
    stepDescription: document.getElementById('step-description'),
    chartPanel: document.getElementById('chart-panel'),
    chartInner: document.getElementById('chart-inner'),
    chartLabel: document.getElementById('chart-label'),
    chartClose: document.getElementById('chart-close'),
};

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
        setFrame(0);

        // Render static analysis panels (lazy — only on first tab visit)
        // Pre-render all immediately in background after a short delay
        setTimeout(() => renderStaticPanels(), 400);

        console.log('Viz loaded. Frames:', state.data.frames.length, '| Nodes:', state.data.topology.nodes.length);
    } catch (err) {
        console.error(err);
        if (els.stepTitle) els.stepTitle.textContent = 'Error loading data';
        if (els.stepDescription) els.stepDescription.innerHTML = 'Run <code style="background:#1e293b;padding:2px 6px;border-radius:4px;color:#60a5fa;">python utils/export_viz_data.py</code> then refresh.';
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
            // Push current frame to 3D
            if (state.data.frames.length) {
                update3DFrame(state.data.frames[state.currentFrame]);
            }
            // Wire auto-rotate toggle
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
}

// ── Tab system ────────────────────────────────────────────────────────────────
function initTabs() {
    const btns = document.querySelectorAll('.tab-btn');
    const panels = document.querySelectorAll('.tab-panel');

    btns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.tab;
            if (tab === state.activeTab) return;
            state.activeTab = tab;

            btns.forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
            panels.forEach(p => p.classList.toggle('active', p.id === `panel-${tab}`));

            // Lazy-render on first visit
            if (state.data) renderStaticPanels();

            // Show/hide playback controls only for graph tab
            const isGraphOrThree = tab === 'graph' || tab === 'three';
            const controlsEl = document.querySelector('.controls');
            if (controlsEl) controlsEl.style.display = (tab === 'graph') ? 'flex' : 'none';

            // Activate / deactivate 3D renderers
            set3DActive(tab === 'three');
            setHebbianActive(tab === 'hebbian');

            // Forward current frame to 3D when switching into 3D tab
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
    els.edgeCount.textContent = state.data.topology.edges.length;

    els.scrubber.addEventListener('input', e => { setFrame(parseInt(e.target.value)); pause(); });
    els.btnPrev.addEventListener('click', () => { setFrame(state.currentFrame - 1); pause(); });
    els.btnNext.addEventListener('click', () => { setFrame(state.currentFrame + 1); pause(); });
    els.btnPlay.addEventListener('click', togglePlay);

    document.addEventListener('keydown', e => {
        if (state.activeTab !== 'graph') return;
        if (e.key === 'ArrowRight') { setFrame(state.currentFrame + 1); pause(); }
        else if (e.key === 'ArrowLeft') { setFrame(state.currentFrame - 1); pause(); }
        else if (e.key === ' ') { e.preventDefault(); togglePlay(); }
        else if (e.key === 'Escape') closeChart();
    });

    if (els.chartClose) els.chartClose.addEventListener('click', closeChart);

    initTabs();

    // Visual intensity toggle
    const intensitySelect = document.getElementById('intensity-select');
    if (intensitySelect) {
        intensitySelect.addEventListener('change', (e) => {
            const level = e.target.value;
            document.body.dataset.intensity = level;
            setIntensity(level);
        });
    }
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
    render();

    if (state.selection && els.chartPanel.classList.contains('visible')) {
        updateCursor(frame.layer);
    }
}

function updateStepPips(activeStep) {
    document.querySelectorAll('.step-pip').forEach((pip, i) => {
        pip.classList.remove('active', 'done');
        if (i === activeStep) pip.classList.add('active');
        else if (i < activeStep) pip.classList.add('done');
    });
    document.querySelectorAll('.step-connector').forEach((conn, i) => {
        conn.classList.remove('done');
        if (i < activeStep) conn.classList.add('done');
    });
}

function updateInfoPanel(frame, stepIdx) {
    if (els.stepBadge) els.stepBadge.textContent = STEP_LABELS[stepIdx] || '';
    if (els.infoPanel) {
        STEP_CLASSES.forEach(c => els.infoPanel.classList.remove(c));
        els.infoPanel.classList.add(STEP_CLASSES[stepIdx] || STEP_CLASSES[0]);
    }
    if (els.stepCounter) els.stepCounter.textContent = `Step ${stepIdx + 1}/4`;
    if (els.stepTitle) els.stepTitle.textContent = frame.step_name || '';
    if (els.stepDescription) els.stepDescription.textContent = frame.description || '';
}

function render() {
    if (!state.data) return;
    const frame = state.data.frames[state.currentFrame];
    updateGraph(frame);
    updateBoard(frame, state.data.config.board_size);
    // Forward to 3D scene if active
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
    if (state.playbackInterval) { clearInterval(state.playbackInterval); state.playbackInterval = null; }
}

loadData();
