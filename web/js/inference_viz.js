// ══════════════════════════════════════════════════════════════════════════════
// inference_viz.js — BDH Architecture Exhaustive Visual Explainer
// Full Training Pipeline + Animated Inference Pop-up
// ══════════════════════════════════════════════════════════════════════════════

// ── Hyperparameters (EXACT — DO NOT SCALE) ──────────────────────────────────
const HP = {
    B: 16, T: 1024, D: 256, L: 6, H: 4,
    N: 32768, Nh: 8192, V: 32000,
};

// ── Color System ─────────────────────────────────────────────────────────────
const COL = {
    // Zone colors
    residual: { bg: '#0c1f4a', border: '#3b82f6', text: '#93c5fd', fill: '#1e40af' },
    neuron:   { bg: '#052e16', border: '#22c55e', text: '#86efac', fill: '#166534' },
    attn:     { bg: '#451a03', border: '#f59e0b', text: '#fde68a', fill: '#92400e' },
    output:   { bg: '#2e1065', border: '#a855f7', text: '#d8b4fe', fill: '#6b21a8' },
    input:    { bg: '#431407', border: '#f97316', text: '#fed7aa', fill: '#9a3412' },
    weight:   { bg: '#1a1a2e', border: '#6366f1', text: '#a5b4fc', fill: '#3730a3' },
    // Generic
    bg: '#09090b', dimBorder: 'rgba(255,255,255,0.06)',
    flash: '#fbbf24', muted: '#52525b', textPrimary: '#e4e4e7',
};

// ── Layout Constants ─────────────────────────────────────────────────────────
const LY = {
    W: 1100,          // SVG width
    CX: 550,          // center X
    blockW: 260,
    blockH: 90,
    bigBlockH: 120,
    arrowGap: 40,     // vertical gap between blocks for arrows
    heatCell: 10,     // mini-heatmap cell size
    zoneGap: 30,
};

// ── Utility ──────────────────────────────────────────────────────────────────
const rand = (lo = -1, hi = 1) => lo + Math.random() * (hi - lo);
const relu = x => Math.max(0, x);
const sleep = ms => new Promise(r => setTimeout(r, ms));
function cscale(lo, hi, dom = [0, 1]) {
    return d3.scaleLinear().domain(dom).range([lo, hi]).clamp(true);
}
function randomHeat(rows, cols, lo = -1, hi = 1) {
    return Array.from({ length: rows }, () =>
        Array.from({ length: cols }, () => lo + Math.random() * (hi - lo)));
}
function sparseHeat(rows, cols, sparsity = 0.95) {
    return Array.from({ length: rows }, () =>
        Array.from({ length: cols }, () => Math.random() > sparsity ? Math.random() * 2 : 0));
}
function causalHeat(n) {
    return Array.from({ length: n }, (_, r) =>
        Array.from({ length: n }, (_, c) => c <= r ? 0.1 + Math.random() * 0.9 : 0));
}

// ── State ────────────────────────────────────────────────────────────────────
let state = {
    svg: null,
    activeFormula: null,
    // Inference popup
    busy: false, cancelAnim: false,
    vt: [], Dx: [], xt: [], S: [], at: [],
    infLayer: 0,
};

// ══════════════════════════════════════════════════════════════════════════════
//  RENDERING PRIMITIVES
// ══════════════════════════════════════════════════════════════════════════════

function renderBlock(parent, cfg) {
    const { x, y, w, h, title, dims, color, id, clickable, subtext } = cfg;
    const g = parent.append('g').attr('id', id || '').attr('class', 'arch-block')
        .attr('transform', `translate(${x - w / 2},${y})`);

    // Background rect
    g.append('rect').attr('width', w).attr('height', h).attr('rx', 10)
        .attr('fill', color.bg).attr('stroke', color.border)
        .attr('stroke-width', 1.5).attr('class', 'block-rect');

    // Glow filter
    if (clickable) {
        g.attr('class', 'arch-block clickable').style('cursor', 'pointer');
        g.append('rect').attr('width', w).attr('height', h).attr('rx', 10)
            .attr('fill', 'none').attr('stroke', color.border)
            .attr('stroke-width', 0).attr('class', 'block-glow')
            .attr('filter', 'url(#glow)');
    }

    // Title
    g.append('text').attr('x', w / 2).attr('y', 22)
        .attr('text-anchor', 'middle').attr('fill', color.text)
        .attr('font-size', '12px').attr('font-weight', '700')
        .attr('font-family', 'Inter, system-ui, sans-serif')
        .text(title);

    // Dimension badge
    if (dims) {
        g.append('text').attr('x', w / 2).attr('y', 40)
            .attr('text-anchor', 'middle').attr('fill', COL.muted)
            .attr('font-size', '10px').attr('font-family', "'IBM Plex Mono', monospace")
            .text(dims);
    }

    // Subtext
    if (subtext) {
        g.append('text').attr('x', w / 2).attr('y', h - 12)
            .attr('text-anchor', 'middle').attr('fill', 'rgba(255,255,255,0.3)')
            .attr('font-size', '9px').attr('font-family', "'IBM Plex Mono', monospace")
            .text(subtext);
    }

    return g;
}

function renderMiniHeatmap(parent, ox, oy, data, rows, cols, colorFn, cellSz = LY.heatCell) {
    const g = parent.append('g').attr('transform', `translate(${ox},${oy})`);
    for (let r = 0; r < rows; r++)
        for (let c = 0; c < cols; c++) {
            g.append('rect')
                .attr('x', c * (cellSz + 1)).attr('y', r * (cellSz + 1))
                .attr('width', cellSz).attr('height', cellSz).attr('rx', 1.5)
                .attr('fill', colorFn(data[r][c]));
        }
    return g;
}

function renderArrow(parent, x1, y1, x2, y2, cfg = {}) {
    const { color = '#3f3f46', label, dashed = false, thick = false } = cfg;
    const g = parent.append('g').attr('class', 'arch-arrow');
    g.append('line')
        .attr('x1', x1).attr('y1', y1).attr('x2', x2).attr('y2', y2)
        .attr('stroke', color).attr('stroke-width', thick ? 2.5 : 1.5)
        .attr('stroke-dasharray', dashed ? '6,4' : 'none')
        .attr('marker-end', `url(#arrow-${color.replace('#', '')})`);
    if (label) {
        const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
        g.append('text').attr('x', mx + 8).attr('y', my + 3)
            .attr('fill', color).attr('font-size', '9px')
            .attr('font-family', "'IBM Plex Mono', monospace")
            .text(label);
    }
    return g;
}

function renderFormula(parent, x, y, cfg) {
    const { html, id, color = COL.textPrimary } = cfg;
    const fo = parent.append('foreignObject')
        .attr('x', x).attr('y', y).attr('width', 400).attr('height', 60)
        .attr('class', 'arch-formula').attr('id', id || '');
    const div = fo.append('xhtml:div')
        .style('font-family', "'Georgia', serif")
        .style('font-size', '12px')
        .style('color', color)
        .style('padding', '4px 10px')
        .style('background', 'rgba(255,255,255,0.02)')
        .style('border', '1px solid rgba(255,255,255,0.06)')
        .style('border-radius', '6px')
        .style('line-height', '1.6')
        .style('opacity', '0.7')
        .style('transition', 'opacity 0.2s, box-shadow 0.2s')
        .html(html);
    return fo;
}

function renderZoneHeader(parent, y, title, color) {
    const g = parent.append('g');
    g.append('line').attr('x1', 40).attr('y1', y).attr('x2', LY.W - 40).attr('y2', y)
        .attr('stroke', color).attr('stroke-width', 1).attr('stroke-dasharray', '8,6')
        .attr('opacity', 0.4);
    g.append('rect').attr('x', LY.CX - 140).attr('y', y - 14).attr('width', 280).attr('height', 28)
        .attr('rx', 14).attr('fill', COL.bg).attr('stroke', color).attr('stroke-width', 1);
    g.append('text').attr('x', LY.CX).attr('y', y + 5).attr('text-anchor', 'middle')
        .attr('fill', color).attr('font-size', '11px').attr('font-weight', '700')
        .attr('letter-spacing', '0.1em').text(title);
    return g;
}

function addArrowMarker(defs, color) {
    const id = 'arrow-' + color.replace('#', '');
    defs.append('marker').attr('id', id)
        .attr('viewBox', '0 0 10 10').attr('refX', 9).attr('refY', 5)
        .attr('markerWidth', 7).attr('markerHeight', 7).attr('orient', 'auto')
        .append('path').attr('d', 'M 0 0 L 10 5 L 0 10 z').attr('fill', color);
}

// ── Residual connection bracket ──────────────────────────────────────────────
function renderResidualSkip(parent, x, y1, y2, color = COL.residual.border) {
    const g = parent.append('g').attr('class', 'residual-skip');
    const cx = x;
    g.append('path')
        .attr('d', `M${cx},${y1} C${cx - 50},${y1} ${cx - 50},${y2} ${cx},${y2}`)
        .attr('fill', 'none').attr('stroke', color)
        .attr('stroke-width', 1.5).attr('stroke-dasharray', '4,4').attr('opacity', 0.5);
    g.append('text').attr('x', cx - 58).attr('y', (y1 + y2) / 2 + 4)
        .attr('text-anchor', 'middle').attr('fill', color).attr('font-size', '9px')
        .attr('font-weight', '600').attr('opacity', 0.6).text('+ residual');
    return g;
}

// ══════════════════════════════════════════════════════════════════════════════
//  STATIC TRAINING PIPELINE
// ══════════════════════════════════════════════════════════════════════════════

function renderStaticPipeline(svg) {
    const scene = svg.append('g').attr('id', 'pipeline-scene');

    // ── Arrow markers ──
    const defs = svg.append('defs');
    addArrowMarker(defs, '#3f3f46');
    addArrowMarker(defs, COL.residual.border);
    addArrowMarker(defs, COL.neuron.border);
    addArrowMarker(defs, COL.attn.border);
    addArrowMarker(defs, COL.output.border);
    addArrowMarker(defs, COL.input.border);
    addArrowMarker(defs, COL.weight.border);
    addArrowMarker(defs, COL.flash);

    // Glow filter
    const glow = defs.append('filter').attr('id', 'glow')
        .attr('x', '-50%').attr('y', '-50%').attr('width', '200%').attr('height', '200%');
    glow.append('feGaussianBlur').attr('stdDeviation', '4').attr('result', 'blur');
    glow.append('feMerge').selectAll('feMergeNode')
        .data(['blur', 'SourceGraphic']).enter()
        .append('feMergeNode').attr('in', d => d);

    let cy = 20; // current y cursor

    // ═══════════════════════════════════════════════════════════════
    //  ZONE 1: INPUT & RESIDUAL STREAM
    // ═══════════════════════════════════════════════════════════════
    renderZoneHeader(scene, cy, 'ZONE 1 — INPUT & RESIDUAL STREAM', COL.residual.border);
    cy += 50;

    // Token IDs
    const tokenBlock = renderBlock(scene, {
        x: LY.CX, y: cy, w: LY.blockW, h: 80, id: 'blk-tokenids',
        title: 'Token IDs', dims: `[B=${HP.B}, T=${HP.T}]`,
        color: COL.input, subtext: `Vocab Size V = ${HP.V.toLocaleString()}`,
    });
    const tokenHeat = randomHeat(3, 10, 0, 1);
    renderMiniHeatmap(tokenBlock, 15, 44, tokenHeat, 3, 10,
        v => d3.interpolateOranges(v), 8);
    cy += 80;

    // Arrow
    renderArrow(scene, LY.CX, cy, LY.CX, cy + LY.arrowGap,
        { color: COL.input.border, label: 'Embedding Lookup' });
    cy += LY.arrowGap;

    // Embedding
    const embBlock = renderBlock(scene, {
        x: LY.CX - 60, y: cy, w: LY.blockW, h: 90, id: 'blk-embedding',
        title: 'Token Embedding', dims: `[${HP.B}, ${HP.T}, D=${HP.D}]`,
        color: COL.residual, subtext: `Embedding Table: [V=${HP.V.toLocaleString()}, D=${HP.D}]`,
    });
    const embHeat = randomHeat(4, 10, -1, 1);
    renderMiniHeatmap(embBlock, 15, 48, embHeat, 4, 10,
        v => cscale('#172554', '#3b82f6', [-1, 1])(v), 8);

    // Positional Encoding (side block)
    const posBlock = renderBlock(scene, {
        x: LY.CX + 220, y: cy + 10, w: 180, h: 70, id: 'blk-posenc',
        title: 'Position Enc', dims: `[T=${HP.T}, D=${HP.D}]`,
        color: { ...COL.residual, bg: '#0e1a3a' },
    });
    // Addition arrow
    renderArrow(scene, LY.CX + 220 - 90, cy + 45, LY.CX - 60 + LY.blockW / 2 + 20, cy + 45,
        { color: COL.residual.border, label: '⊕ add' });

    cy += 90;

    // Arrow down to residual stream
    renderArrow(scene, LY.CX, cy, LY.CX, cy + 30, { color: COL.residual.border });
    cy += 30;

    // Residual Stream banner
    const resY = cy;
    scene.append('rect').attr('x', 80).attr('y', cy).attr('width', LY.W - 160).attr('height', 36)
        .attr('rx', 18).attr('fill', COL.residual.bg)
        .attr('stroke', COL.residual.border).attr('stroke-width', 1.5)
        .attr('stroke-dasharray', '6,4');
    scene.append('text').attr('x', LY.CX).attr('y', cy + 22).attr('text-anchor', 'middle')
        .attr('fill', COL.residual.text).attr('font-size', '11px').attr('font-weight', '700')
        .attr('letter-spacing', '0.06em')
        .text(`═══ RESIDUAL STREAM   [B=${HP.B}, T=${HP.T}, D=${HP.D}]  ═══`);
    cy += 36 + LY.zoneGap;

    // ═══════════════════════════════════════════════════════════════
    //  ZONE 2: INSIDE A SINGLE BDH LAYER ("THE BREATHING CYCLE")
    // ═══════════════════════════════════════════════════════════════
    renderZoneHeader(scene, cy, `ZONE 2 — BDH LAYER (×${HP.L} Layers) — THE BREATHING CYCLE`, COL.neuron.border);
    cy += 50;

    // Layer wrapper rect (subtle border around entire zone 2)
    const z2startY = cy;

    // A. v_in
    const vinY = cy;
    renderBlock(scene, {
        x: LY.CX, y: cy, w: LY.blockW, h: 70, id: 'blk-vin',
        title: 'v_in (Input Residual)', dims: `[B=${HP.B}, T=${HP.T}, D=${HP.D}]`,
        color: COL.residual,
    });
    // Formula for v_in → D_x
    renderFormula(scene, LY.CX + 160, cy + 5, {
        id: 'f-expand',
        html: `<span style="color:${COL.neuron.text}">x</span> = Dropout(ReLU(<span style="color:${COL.residual.text}">v<sub>in</sub></span> @ <span style="color:${COL.weight.text}">D<sub>x</sub></span>))`,
    });
    cy += 70;

    renderArrow(scene, LY.CX, cy, LY.CX, cy + LY.arrowGap,
        { color: COL.residual.border, label: `matmul  [B,T,D]×[H,D,Nh]` });
    cy += LY.arrowGap;

    // B. Decoder D_x
    renderBlock(scene, {
        x: LY.CX, y: cy, w: 300, h: LY.bigBlockH, id: 'blk-dx',
        title: 'Decoder D_x (Expansion)', dims: `[H=${HP.H}, D=${HP.D}, N_h=${HP.Nh.toLocaleString()}]`,
        color: COL.weight, subtext: `${HP.H} heads × ${HP.D}×${HP.Nh.toLocaleString()} = ${(HP.H * HP.D * HP.Nh / 1e6).toFixed(1)}M params`,
    });
    const dxHeat = randomHeat(6, 16, -1, 1);
    renderMiniHeatmap(scene.select('#blk-dx'), 15, 50, dxHeat, 6, 16,
        v => cscale('#1e1b4b', '#6366f1', [-1, 1])(v), 7);
    cy += LY.bigBlockH;

    renderArrow(scene, LY.CX, cy, LY.CX, cy + 35,
        { color: COL.neuron.border, label: '→ NEURON SPACE' });
    cy += 35;

    // C. Active Neurons x
    renderBlock(scene, {
        x: LY.CX, y: cy, w: 320, h: 110, id: 'blk-x',
        title: 'Active Neurons (x) — SPARSE', dims: `[B=${HP.B}, H=${HP.H}, T=${HP.T}, N_h=${HP.Nh.toLocaleString()}]`,
        color: COL.neuron, subtext: '~95% zeros after ReLU — brain-like sparse code',
    });
    const xHeat = sparseHeat(5, 20, 0.95);
    renderMiniHeatmap(scene.select('#blk-x'), 15, 50, xHeat, 5, 20,
        v => v <= 0 ? '#0a0a0f' : d3.interpolateGreens(Math.min(v, 1)), 7);
    // Sparsity label
    renderFormula(scene, LY.CX + 190, cy + 10, {
        id: 'f-sparse',
        html: `<span style="color:#fbbf24">~95% sparsity</span><br><span style="font-size:10px;color:#52525b">Only ~${Math.round(HP.Nh * 0.05)} of ${HP.Nh.toLocaleString()} neurons fire</span>`,
    });
    cy += 110;

    renderArrow(scene, LY.CX, cy, LY.CX, cy + 35,
        { color: COL.neuron.border });
    cy += 35;

    // D. Attention Core — THE POP-UP TRIGGER
    const attnG = renderBlock(scene, {
        x: LY.CX, y: cy, w: 320, h: 160, id: 'blk-attn',
        title: '⚡ ATTENTION CORE', dims: 'CLICK FOR INFERENCE MODE →',
        color: COL.attn, clickable: true,
        subtext: 'Training: Parallel  |  Inference: Recurrent O(1)',
    });
    // Sub-formulas inside
    const attnInner = scene.select('#blk-attn');
    attnInner.append('text').attr('x', 160).attr('y', 70).attr('text-anchor', 'middle')
        .attr('fill', COL.attn.text).attr('font-size', '10px')
        .attr('font-family', "'Georgia', serif")
        .text(`Scores = x @ xᵀ   [${HP.B},${HP.H},${HP.T},${HP.T}]`);
    attnInner.append('text').attr('x', 160).attr('y', 88).attr('text-anchor', 'middle')
        .attr('fill', COL.attn.text).attr('font-size', '10px')
        .attr('font-family', "'Georgia', serif")
        .text(`a_out = Scores @ v_in   [${HP.B},${HP.H},${HP.T},${HP.D}]`);
    // Causal mask heatmap
    const causalData = causalHeat(8);
    renderMiniHeatmap(attnInner, 20, 96, causalData, 8, 8,
        v => v <= 0 ? '#1a1a1a' : d3.interpolateYlOrRd(v * 0.7 + 0.3), 6);
    attnInner.append('text').attr('x', 90).attr('y', 108)
        .attr('fill', '#52525b').attr('font-size', '8px').text('Causal Mask');

    // Formulas (right side)
    renderFormula(scene, LY.CX + 190, cy + 5, {
        id: 'f-attn1',
        html: `<strong style="color:${COL.attn.text}">Training Mode:</strong><br>
            Scores = <span style="color:${COL.neuron.text}">x</span> @ <span style="color:${COL.neuron.text}">x</span>.mT → [${HP.B},${HP.H},${HP.T},${HP.T}]<br>
            <span style="color:${COL.residual.text}">a<sub>out</sub></span> = Scores @ <span style="color:${COL.residual.text}">v<sub>in</sub></span> → [${HP.B},${HP.H},${HP.T},${HP.D}]`,
    });

    // Click handler for inference popup
    attnG.on('click', () => openInferencePopup());
    cy += 160;

    renderArrow(scene, LY.CX, cy, LY.CX, cy + 35,
        { color: COL.attn.border, label: `a_out [${HP.B},${HP.H},${HP.T},${HP.D}]` });
    cy += 35;

    // E. Projection Decoder D_y
    renderBlock(scene, {
        x: LY.CX, y: cy, w: 320, h: LY.bigBlockH, id: 'blk-dy',
        title: 'Projection Decoder D_y', dims: `[H=${HP.H}, D=${HP.D}, N_h=${HP.Nh.toLocaleString()}]`,
        color: COL.weight, subtext: `y_attn = (ReLU(LN(a_out) @ D_y)) ⊙ x`,
    });
    const dyHeat = randomHeat(6, 16, -1, 1);
    renderMiniHeatmap(scene.select('#blk-dy'), 15, 50, dyHeat, 6, 16,
        v => cscale('#1e1b4b', '#6366f1', [-1, 1])(v), 7);
    renderFormula(scene, LY.CX + 190, cy + 5, {
        id: 'f-dy',
        html: `<span style="color:${COL.neuron.text}">y<sub>attn</sub></span> = ReLU(LN(<span style="color:${COL.residual.text}">a<sub>out</sub></span>) @ <span style="color:${COL.weight.text}">D<sub>y</sub></span>) <span style="color:#fbbf24">⊙</span> <span style="color:${COL.neuron.text}">x</span><br>
            <span style="font-size:10px;color:#52525b">Element-wise gating with original x → [${HP.B},${HP.H},${HP.T},${HP.Nh.toLocaleString()}]</span>`,
    });
    cy += LY.bigBlockH;

    renderArrow(scene, LY.CX, cy, LY.CX, cy + 35,
        { color: COL.neuron.border, label: `y_attn [${HP.B},${HP.H},${HP.T},${HP.Nh.toLocaleString()}]` });
    cy += 35;

    // F. Reshape
    renderBlock(scene, {
        x: LY.CX, y: cy, w: LY.blockW, h: 65, id: 'blk-reshape',
        title: 'Reshape (Heads → Brain)',
        dims: `[${HP.B}, 1, ${HP.T}, N=${HP.N.toLocaleString()}]`,
        color: { ...COL.neuron, bg: '#071f0e' },
        subtext: `${HP.H} heads × ${HP.Nh.toLocaleString()} = ${HP.N.toLocaleString()} Brain neurons`,
    });
    cy += 65;

    renderArrow(scene, LY.CX, cy, LY.CX, cy + 30,
        { color: COL.neuron.border, label: '→ COMPRESS BACK' });
    cy += 30;

    // G. Encoder E
    renderBlock(scene, {
        x: LY.CX, y: cy, w: 300, h: LY.bigBlockH, id: 'blk-encoder',
        title: 'Encoder E (Compression)', dims: `[N=${HP.N.toLocaleString()}, D=${HP.D}]`,
        color: COL.weight, subtext: `${(HP.N * HP.D / 1e6).toFixed(1)}M params — compresses brain → latent`,
    });
    const eHeat = randomHeat(10, 6, -1, 1);
    renderMiniHeatmap(scene.select('#blk-encoder'), 15, 50, eHeat, 10, 6,
        v => cscale('#1e1b4b', '#6366f1', [-1, 1])(v), 7);
    cy += LY.bigBlockH;

    renderArrow(scene, LY.CX, cy, LY.CX, cy + 30,
        { color: COL.residual.border, label: '→ LATENT SPACE' });
    cy += 30;

    // H. v_out with residual connection
    const voutY = cy;
    renderBlock(scene, {
        x: LY.CX, y: cy, w: LY.blockW, h: 85, id: 'blk-vout',
        title: 'v_out (Output Residual)', dims: `[B=${HP.B}, T=${HP.T}, D=${HP.D}]`,
        color: COL.residual,
    });
    renderFormula(scene, LY.CX + 160, cy + 5, {
        id: 'f-vout',
        html: `<span style="color:${COL.residual.text}">v<sub>out</sub></span> = <span style="color:${COL.residual.text}">v<sub>in</sub></span> + Dropout(LN(Brain @ <span style="color:${COL.weight.text}">E</span>))`,
    });

    // Residual skip connection line
    renderResidualSkip(scene, LY.CX - LY.blockW / 2 - 5, vinY + 35, cy + 42);
    cy += 85;

    // Layer Repeat indicator
    cy += 15;
    const z2endY = cy;
    scene.append('rect')
        .attr('x', LY.CX - 200).attr('y', z2startY - 15)
        .attr('width', 400).attr('height', z2endY - z2startY + 30)
        .attr('rx', 16).attr('fill', 'none')
        .attr('stroke', COL.neuron.border).attr('stroke-width', 1)
        .attr('stroke-dasharray', '8,6').attr('opacity', 0.15);

    // "× 6 Layers" badge bottom
    scene.append('rect').attr('x', LY.CX - 60).attr('y', cy - 2)
        .attr('width', 120).attr('height', 28).attr('rx', 14)
        .attr('fill', COL.neuron.fill).attr('stroke', COL.neuron.border).attr('stroke-width', 1.5);
    scene.append('text').attr('x', LY.CX).attr('y', cy + 17).attr('text-anchor', 'middle')
        .attr('fill', COL.neuron.text).attr('font-size', '12px').attr('font-weight', '800')
        .attr('letter-spacing', '0.08em').text(`× ${HP.L} LAYERS`);
    cy += 36 + LY.zoneGap;

    // ═══════════════════════════════════════════════════════════════
    //  ZONE 3: OUTPUT & READOUT
    // ═══════════════════════════════════════════════════════════════
    renderZoneHeader(scene, cy, 'ZONE 3 — OUTPUT & READOUT', COL.output.border);
    cy += 50;

    // Final v_out
    renderBlock(scene, {
        x: LY.CX, y: cy, w: LY.blockW, h: 70, id: 'blk-final-vout',
        title: `Final v_out (after L=${HP.L} layers)`, dims: `[${HP.B}, ${HP.T}, D=${HP.D}]`,
        color: COL.residual,
    });
    cy += 70;

    renderArrow(scene, LY.CX, cy, LY.CX, cy + 35,
        { color: COL.output.border, label: `matmul  [B,T,D]×[D,V]` });
    cy += 35;

    // Readout Matrix
    renderBlock(scene, {
        x: LY.CX, y: cy, w: 300, h: LY.bigBlockH, id: 'blk-readout',
        title: 'Readout (Unembedding)', dims: `[D=${HP.D}, V=${HP.V.toLocaleString()}]`,
        color: COL.output, subtext: `${(HP.D * HP.V / 1e6).toFixed(1)}M params`,
    });
    const readHeat = randomHeat(6, 16, -1, 1);
    renderMiniHeatmap(scene.select('#blk-readout'), 15, 50, readHeat, 6, 16,
        v => cscale('#2e1065', '#a855f7', [-1, 1])(v), 7);
    cy += LY.bigBlockH;

    renderArrow(scene, LY.CX, cy, LY.CX, cy + 35,
        { color: COL.output.border, label: 'softmax' });
    cy += 35;

    // Logits
    renderBlock(scene, {
        x: LY.CX, y: cy, w: LY.blockW, h: 80, id: 'blk-logits',
        title: 'Logits (Output)', dims: `[B=${HP.B}, T=${HP.T}, V=${HP.V.toLocaleString()}]`,
        color: COL.output, subtext: `${(HP.B * HP.T * HP.V / 1e9).toFixed(2)} billion values per forward pass`,
    });
    cy += 80 + 40;

    return cy; // total height
}

// ══════════════════════════════════════════════════════════════════════════════
//  FORMULA HIGHLIGHT SYSTEM
// ══════════════════════════════════════════════════════════════════════════════

function setupHighlights(svg) {
    svg.selectAll('.arch-block.clickable')
        .on('mouseenter', function () {
            d3.select(this).select('.block-glow')
                .transition().duration(200).attr('stroke-width', 3).attr('opacity', 0.8);
            d3.select(this).select('.block-rect')
                .transition().duration(200).attr('stroke-width', 2.5);
        })
        .on('mouseleave', function () {
            d3.select(this).select('.block-glow')
                .transition().duration(300).attr('stroke-width', 0).attr('opacity', 0);
            d3.select(this).select('.block-rect')
                .transition().duration(300).attr('stroke-width', 1.5);
        });

    // Hover on any block highlights its associated formula
    const pairs = [
        ['blk-dx', 'f-expand'], ['blk-x', 'f-sparse'],
        ['blk-attn', 'f-attn1'], ['blk-dy', 'f-dy'], ['blk-vout', 'f-vout'],
    ];
    pairs.forEach(([blkId, fId]) => {
        const blk = svg.select(`#${blkId}`);
        const formula = svg.select(`#${fId}`);
        if (blk.empty() || formula.empty()) return;
        blk.on('mouseenter.formula', () => {
            formula.select('div')
                .style('opacity', '1')
                .style('box-shadow', '0 0 15px rgba(99,102,241,0.3)');
        }).on('mouseleave.formula', () => {
            formula.select('div')
                .style('opacity', '0.7')
                .style('box-shadow', 'none');
        });
    });
}

// ══════════════════════════════════════════════════════════════════════════════
//  INFERENCE POPUP (RECURRENT MODE)
// ══════════════════════════════════════════════════════════════════════════════

// Toy dimensions for animation (representative)
const TOY = { D: 4, N: 12 };
const TCELL = 36, TGAP = 2, TSTP = TCELL + TGAP;
const ANIM_CELL = 100;

const orangeScale = cscale('#78350f', '#f97316', [-1, 1]);
const grayScale = cscale('#27272a', '#a1a1aa', [-1, 1]);
const purpleScale = cscale('#3b0764', '#a855f7', [-1, 1]);
function greenOrDark(v) {
    return v <= 0 ? '#0a0a0f' : d3.interpolateRgb('#064e3b', '#22c55e')(Math.min(v / 1.5, 1));
}

function initInferenceData() {
    state.vt = Array.from({ length: TOY.D }, () => +rand(-1, 1).toFixed(2));
    state.Dx = Array.from({ length: TOY.D }, () =>
        Array.from({ length: TOY.N }, () => +rand(-1, 1).toFixed(2)));
    if (state.S.length === 0)
        state.S = Array.from({ length: TOY.N }, () =>
            Array.from({ length: TOY.D }, () => +(Math.random() * 0.15).toFixed(3)));
    state.xt = new Array(TOY.N).fill(0);
    state.at = new Array(TOY.D).fill(0);
}

function computeExpansion() {
    const raw = new Array(TOY.N).fill(0);
    for (let j = 0; j < TOY.N; j++) {
        let sum = 0;
        for (let i = 0; i < TOY.D; i++) sum += state.vt[i] * state.Dx[i][j];
        raw[j] = sum;
    }
    const threshold = d3.quantile(raw.map(Math.abs).sort(d3.ascending), 0.65);
    for (let j = 0; j < TOY.N; j++)
        state.xt[j] = raw[j] > threshold ? +relu(raw[j]).toFixed(3) : 0;
}
function computeRead() {
    for (let d = 0; d < TOY.D; d++) {
        let sum = 0;
        for (let n = 0; n < TOY.N; n++) sum += state.xt[n] * state.S[n][d];
        state.at[d] = +sum.toFixed(3);
    }
}
function computeWrite() {
    for (let n = 0; n < TOY.N; n++)
        for (let d = 0; d < TOY.D; d++)
            state.S[n][d] = +(state.S[n][d] + state.xt[n] * state.vt[d]).toFixed(4);
}

function drawStaticMat(g, data, rows, cols, colorFn, ox, oy, label) {
    const mg = g.append('g').attr('transform', `translate(${ox},${oy})`);
    mg.append('text').attr('class', 'iv-label')
        .attr('x', (cols * TSTP) / 2).attr('y', -10)
        .attr('text-anchor', 'middle').text(label);
    for (let r = 0; r < rows; r++)
        for (let c = 0; c < cols; c++) {
            const v = Array.isArray(data[0]) ? data[r][c] : data[r * cols + c];
            mg.append('rect').attr('x', c * TSTP).attr('y', r * TSTP)
                .attr('width', TCELL).attr('height', TCELL).attr('rx', 3)
                .attr('fill', colorFn(v))
                .attr('stroke', COL.dimBorder).attr('stroke-width', 1)
                .attr('data-r', r).attr('data-c', c);
            if (rows * cols <= 64)
                mg.append('text').attr('x', c * TSTP + TCELL / 2).attr('y', r * TSTP + TCELL / 2 + 4)
                    .attr('text-anchor', 'middle').attr('fill', 'rgba(255,255,255,0.7)')
                    .attr('font-size', '8px').attr('font-family', 'monospace')
                    .text(typeof v === 'number' ? v.toFixed(1) : '');
        }
    return mg;
}

function drawEmptyRes(g, rows, cols, ox, oy, label) {
    const mg = g.append('g').attr('class', 'iv-result-group')
        .attr('transform', `translate(${ox},${oy})`);
    mg.append('text').attr('class', 'iv-label')
        .attr('x', (cols * TSTP) / 2).attr('y', -10)
        .attr('text-anchor', 'middle').text(label);
    for (let r = 0; r < rows; r++)
        for (let c = 0; c < cols; c++) {
            mg.append('rect').attr('class', 'iv-res-cell')
                .attr('x', c * TSTP).attr('y', r * TSTP)
                .attr('width', TCELL).attr('height', TCELL).attr('rx', 3)
                .attr('fill', '#111116')
                .attr('stroke', 'rgba(255,255,255,0.08)').attr('stroke-width', 1)
                .attr('data-r', r).attr('data-c', c);
            mg.append('text').attr('class', 'iv-res-val')
                .attr('x', c * TSTP + TCELL / 2).attr('y', r * TSTP + TCELL / 2 + 4)
                .attr('text-anchor', 'middle').attr('fill', 'rgba(255,255,255,0.3)')
                .attr('font-size', '8px').attr('font-family', 'monospace')
                .attr('data-r', r).attr('data-c', c).text('·');
        }
    return mg;
}

async function animateMatmul(leftG, rightG, resultG, result, rows, cols, colorFn, perCell = ANIM_CELL) {
    for (let r = 0; r < rows; r++) {
        for (let cc = 0; cc < cols; cc++) {
            if (state.cancelAnim) return;
            leftG.selectAll('rect').attr('stroke', COL.dimBorder).attr('stroke-width', 1);
            leftG.selectAll('rect')
                .filter(function () { return +this.getAttribute('data-r') === r; })
                .attr('stroke', COL.flash).attr('stroke-width', 2.5);
            rightG.selectAll('rect').attr('stroke', COL.dimBorder).attr('stroke-width', 1);
            rightG.selectAll('rect')
                .filter(function () { return +this.getAttribute('data-c') === cc; })
                .attr('stroke', COL.flash).attr('stroke-width', 2.5);
            await sleep(perCell * 0.4);
            if (state.cancelAnim) return;
            const val = Array.isArray(result[0]) ? result[r][cc] : result[r * cols + cc];
            resultG.selectAll('rect.iv-res-cell')
                .filter(function () {
                    return +this.getAttribute('data-r') === r && +this.getAttribute('data-c') === cc;
                })
                .transition().duration(perCell * 0.5)
                .attr('fill', colorFn(val)).attr('stroke', COL.flash).attr('stroke-width', 2);
            resultG.selectAll('text.iv-res-val')
                .filter(function () {
                    return +this.getAttribute('data-r') === r && +this.getAttribute('data-c') === cc;
                })
                .text(typeof val === 'number' ? val.toFixed(2) : '')
                .attr('fill', 'rgba(255,255,255,0.85)');
            await sleep(perCell * 0.6);
        }
    }
    leftG.selectAll('rect').attr('stroke', COL.dimBorder).attr('stroke-width', 1);
    rightG.selectAll('rect').attr('stroke', COL.dimBorder).attr('stroke-width', 1);
    resultG.selectAll('rect.iv-res-cell')
        .transition().duration(400).attr('stroke', 'rgba(255,255,255,0.12)').attr('stroke-width', 1);
}

async function animateOuterProduct(leftG, rightG, resultG, result, rows, cols, colorFn, perRow = 160) {
    for (let r = 0; r < rows; r++) {
        if (state.cancelAnim) return;
        leftG.selectAll('rect').attr('stroke', COL.dimBorder).attr('stroke-width', 1);
        leftG.selectAll('rect')
            .filter(function () { return +this.getAttribute('data-r') === r; })
            .attr('stroke', COL.flash).attr('stroke-width', 2.5);
        rightG.selectAll('rect').attr('stroke', COL.flash).attr('stroke-width', 2);
        await sleep(perRow * 0.3);
        if (state.cancelAnim) return;
        for (let cc = 0; cc < cols; cc++) {
            const val = result[r][cc];
            resultG.selectAll('rect.iv-res-cell')
                .filter(function () {
                    return +this.getAttribute('data-r') === r && +this.getAttribute('data-c') === cc;
                })
                .transition().duration(perRow * 0.4)
                .attr('fill', colorFn(val)).attr('stroke', COL.flash).attr('stroke-width', 2);
            resultG.selectAll('text.iv-res-val')
                .filter(function () {
                    return +this.getAttribute('data-r') === r && +this.getAttribute('data-c') === cc;
                })
                .text(typeof val === 'number' ? val.toFixed(2) : '')
                .attr('fill', 'rgba(255,255,255,0.85)');
        }
        await sleep(perRow * 0.7);
    }
    leftG.selectAll('rect').attr('stroke', COL.dimBorder).attr('stroke-width', 1);
    rightG.selectAll('rect').attr('stroke', COL.dimBorder).attr('stroke-width', 1);
    resultG.selectAll('rect.iv-res-cell')
        .transition().duration(400).attr('stroke', 'rgba(255,255,255,0.12)').attr('stroke-width', 1);
}

// ── Inference Popup Open ─────────────────────────────────────────────────────
function openInferencePopup() {
    if (state.busy) return;
    state.busy = true;
    state.cancelAnim = false;
    initInferenceData();

    const overlay = document.getElementById('iv-overlay');
    const modal = document.getElementById('iv-modal');
    overlay.classList.add('active');
    modal.classList.add('active');

    const body = document.getElementById('iv-modal-body');
    body.innerHTML = '';

    // Build the inference scene
    const dimInfo = document.createElement('div');
    dimInfo.className = 'inf-dim-info';
    dimInfo.innerHTML = `
        <div class="inf-dim-row">
            <span class="inf-dim-chip blue">v<sub>t</sub> [1, D=${HP.D}]</span>
            <span class="inf-dim-chip green">x<sub>t</sub> [1, N<sub>h</sub>=${HP.Nh.toLocaleString()}]</span>
            <span class="inf-dim-chip purple">a<sub>t</sub> [1, D=${HP.D}]</span>
            <span class="inf-dim-chip amber">S [N<sub>h</sub>=${HP.Nh.toLocaleString()}, D=${HP.D}]</span>
        </div>
        <div class="inf-dim-note">Representative slice below — actual dimensions in brackets above</div>
    `;
    body.appendChild(dimInfo);

    // S matrix (always visible)
    const sSection = document.createElement('div');
    sSection.className = 'inf-s-section';
    sSection.innerHTML = `<div class="inf-s-title">THE FIXED STATE MATRIX S — accumulates ALL past context</div>
        <div class="inf-s-note">[N<sub>h</sub>=${HP.Nh.toLocaleString()}, D=${HP.D}] — represents ${HP.Nh.toLocaleString()} × ${HP.D} = ${(HP.Nh * HP.D / 1e6).toFixed(1)}M values. Does NOT exist as persistent object in training.</div>
        <div id="inf-s-viz"></div>`;
    body.appendChild(sSection);

    // Render S matrix
    const sMax = Math.max(0.01, ...state.S.flat().map(Math.abs));
    const dynBlue = cscale(COL.residual.bg, COL.residual.border, [0, sMax]);
    const sSvg = d3.select('#inf-s-viz').append('svg')
        .attr('viewBox', `0 0 ${TOY.D * TSTP + 70} ${TOY.N * TSTP + 30}`)
        .attr('width', '100%').attr('preserveAspectRatio', 'xMidYMid meet')
        .style('max-width', '240px').style('background', '#0a0a12').style('border-radius', '8px');
    drawStaticMat(sSvg, state.S, TOY.N, TOY.D, v => dynBlue(Math.abs(v)),
        20, 20, `S [${HP.Nh.toLocaleString()}×${HP.D}]`);

    // Animation steps container
    const stepsDiv = document.createElement('div');
    stepsDiv.className = 'inf-steps';
    stepsDiv.innerHTML = `
        <div class="inf-step-header">INFERENCE CYCLE — Layer ${state.infLayer + 1}, Head 1, Step t</div>
        <div id="inf-step1" class="inf-step-box">
            <div class="inf-step-title" style="color:${COL.input.border}">Step 1 — EXPAND</div>
            <div class="inf-step-formula">x<sub>t</sub> = ReLU(v<sub>t</sub> [1,${HP.D}] × D<sub>x</sub> [${HP.D},${HP.Nh.toLocaleString()}]) → sparse x<sub>t</sub> [1,${HP.Nh.toLocaleString()}]</div>
            <div id="inf-step1-svg"></div>
            <div id="inf-step1-status" class="inf-step-status">Waiting...</div>
        </div>
        <div id="inf-step2" class="inf-step-box">
            <div class="inf-step-title" style="color:${COL.residual.border}">Step 2 — READ MEMORY</div>
            <div class="inf-step-formula">a<sub>t</sub> = x<sub>t</sub> [1,${HP.Nh.toLocaleString()}] × S [${HP.Nh.toLocaleString()},${HP.D}] → a<sub>t</sub> [1,${HP.D}]</div>
            <div id="inf-step2-svg"></div>
            <div id="inf-step2-status" class="inf-step-status">Waiting...</div>
        </div>
        <div id="inf-step3" class="inf-step-box">
            <div class="inf-step-title" style="color:${COL.flash}">Step 3 — WRITE MEMORY (Hebbian Update)</div>
            <div class="inf-step-formula">ΔS = x<sub>t</sub><sup>T</sup> [${HP.Nh.toLocaleString()},1] × v<sub>t</sub> [1,${HP.D}] → [${HP.Nh.toLocaleString()},${HP.D}]<br>S<sub>new</sub> = S<sub>old</sub> + ΔS</div>
            <div id="inf-step3-svg"></div>
            <div id="inf-step3-status" class="inf-step-status">Waiting...</div>
        </div>
    `;
    body.appendChild(stepsDiv);

    // Auto-play button
    const playBtn = document.createElement('button');
    playBtn.className = 'inf-play-btn';
    playBtn.textContent = '▶ Auto-Play Full Cycle';
    body.appendChild(playBtn);

    playBtn.addEventListener('click', () => runFullInferenceCycle());

    // Close handler
    const closeFn = () => {
        state.cancelAnim = true;
        overlay.classList.remove('active');
        modal.classList.remove('active');
        state.busy = false;
    };
    document.getElementById('iv-modal-close').onclick = closeFn;
    overlay.onclick = (e) => { if (e.target === overlay) closeFn(); };
}

async function runFullInferenceCycle() {
    // Step 1: Expand
    const s1Status = document.getElementById('inf-step1-status');
    const s1Wrap = document.getElementById('inf-step1-svg');
    s1Wrap.innerHTML = '';
    s1Status.textContent = 'Animating...';
    s1Status.className = 'inf-step-status animating';

    computeExpansion();
    const opGap = 40;
    const leftW = TOY.D * TSTP, rightW = TOY.N * TSTP, resW = TOY.N * TSTP;
    const s1TotW = leftW + opGap + rightW + opGap + resW + 20;
    const s1TotH = TOY.D * TSTP + 50;
    const s1Svg = d3.select(s1Wrap).append('svg')
        .attr('viewBox', `0 0 ${s1TotW} ${s1TotH}`)
        .attr('width', '100%').attr('preserveAspectRatio', 'xMidYMid meet')
        .style('background', '#0c0c12').style('border-radius', '8px');
    const oY = 25;
    const lG = drawStaticMat(s1Svg, [state.vt], 1, TOY.D, v => orangeScale(v), 10, oY, 'v_t');
    s1Svg.append('text').attr('x', 10 + leftW + opGap / 2).attr('y', oY + TSTP / 2 + 5)
        .attr('text-anchor', 'middle').attr('fill', '#71717a').attr('font-size', '18px').attr('font-weight', '700').text('×');
    const rG = drawStaticMat(s1Svg, state.Dx, TOY.D, TOY.N, v => grayScale(v), 10 + leftW + opGap, oY, 'D_x');
    s1Svg.append('text').attr('x', 10 + leftW + opGap + rightW + opGap / 2).attr('y', oY + TSTP / 2 + 5)
        .attr('text-anchor', 'middle').attr('fill', '#71717a').attr('font-size', '18px').attr('font-weight', '700').text('=');
    const resG = drawEmptyRes(s1Svg, 1, TOY.N, 10 + leftW + opGap + rightW + opGap, oY, 'x_t');

    await sleep(300);
    await animateMatmul(lG, rG, resG, [state.xt], 1, TOY.N, greenOrDark, ANIM_CELL);

    const active = state.xt.filter(v => v > 0).length;
    s1Status.textContent = `✓ ${active}/${TOY.N} neurons active — ${Math.round(((TOY.N - active) / TOY.N) * 100)}% sparsity`;
    s1Status.className = 'inf-step-status done';

    if (state.cancelAnim) return;
    await sleep(600);

    // Step 2: Read
    const s2Status = document.getElementById('inf-step2-status');
    const s2Wrap = document.getElementById('inf-step2-svg');
    s2Wrap.innerHTML = '';
    s2Status.textContent = 'Animating...';
    s2Status.className = 'inf-step-status animating';

    computeRead();
    const l2W = TOY.N * TSTP, r2W = TOY.D * TSTP, r2H = TOY.N * TSTP, res2W = TOY.D * TSTP;
    const s2TotW = l2W + opGap + r2W + opGap + res2W + 20;
    const s2TotH = r2H + 50;
    const s2Svg = d3.select(s2Wrap).append('svg')
        .attr('viewBox', `0 0 ${s2TotW} ${s2TotH}`)
        .attr('width', '100%').attr('preserveAspectRatio', 'xMidYMid meet')
        .style('background', '#0c0c12').style('border-radius', '8px');
    const sMax2 = Math.max(0.01, ...state.S.flat().map(Math.abs));
    const dynB2 = cscale(COL.residual.bg, COL.residual.border, [0, sMax2]);
    const lG2 = drawStaticMat(s2Svg, [state.xt], 1, TOY.N, greenOrDark, 10, oY, 'x_t');
    s2Svg.append('text').attr('x', 10 + l2W + opGap / 2).attr('y', oY + TSTP / 2 + 5)
        .attr('text-anchor', 'middle').attr('fill', '#71717a').attr('font-size', '18px').attr('font-weight', '700').text('×');
    const rG2 = drawStaticMat(s2Svg, state.S, TOY.N, TOY.D, v => dynB2(Math.abs(v)), 10 + l2W + opGap, oY, 'S');
    s2Svg.append('text').attr('x', 10 + l2W + opGap + r2W + opGap / 2).attr('y', oY + TSTP / 2 + 5)
        .attr('text-anchor', 'middle').attr('fill', '#71717a').attr('font-size', '18px').attr('font-weight', '700').text('=');
    const resG2 = drawEmptyRes(s2Svg, 1, TOY.D, 10 + l2W + opGap + r2W + opGap, oY, 'a_t');

    await sleep(300);
    await animateMatmul(lG2, rG2, resG2, [state.at], 1, TOY.D, v => purpleScale(v), ANIM_CELL * 2);

    s2Status.textContent = `✓ Output a_t retrieved — max |a| = ${Math.max(...state.at.map(Math.abs)).toFixed(3)}`;
    s2Status.className = 'inf-step-status done';

    if (state.cancelAnim) return;
    await sleep(600);

    // Step 3: Write (Hebbian)
    const s3Status = document.getElementById('inf-step3-status');
    const s3Wrap = document.getElementById('inf-step3-svg');
    s3Wrap.innerHTML = '';
    s3Status.textContent = 'Animating...';
    s3Status.className = 'inf-step-status animating';

    const deltaS = Array.from({ length: TOY.N }, (_, n) =>
        Array.from({ length: TOY.D }, (_, d) => +(state.xt[n] * state.vt[d]).toFixed(4)));

    const l3W = 1 * TSTP, l3H = TOY.N * TSTP, r3W = TOY.D * TSTP;
    const res3W = TOY.D * TSTP, res3H = TOY.N * TSTP;
    const s3TotW = l3W + opGap + r3W + opGap + res3W + 20;
    const s3TotH = Math.max(l3H, res3H) + 50;
    const s3Svg = d3.select(s3Wrap).append('svg')
        .attr('viewBox', `0 0 ${s3TotW} ${s3TotH}`)
        .attr('width', '100%').attr('preserveAspectRatio', 'xMidYMid meet')
        .style('background', '#0c0c12').style('border-radius', '8px');
    const xtCol = state.xt.map(v => [v]);
    const lG3 = drawStaticMat(s3Svg, xtCol, TOY.N, 1, greenOrDark, 10, oY, 'x_tᵀ');
    s3Svg.append('text').attr('x', 10 + l3W + opGap / 2).attr('y', oY + l3H / 2 + 5)
        .attr('text-anchor', 'middle').attr('fill', '#71717a').attr('font-size', '18px').attr('font-weight', '700').text('×');
    const rG3 = drawStaticMat(s3Svg, [state.vt], 1, TOY.D, v => orangeScale(v), 10 + l3W + opGap, oY, 'v_t');
    s3Svg.append('text').attr('x', 10 + l3W + opGap + r3W + opGap / 2).attr('y', oY + l3H / 2 + 5)
        .attr('text-anchor', 'middle').attr('fill', '#71717a').attr('font-size', '18px').attr('font-weight', '700').text('=');
    const dsMax = Math.max(0.01, ...deltaS.flat().map(Math.abs));
    const dsScale = cscale('#1a1a2e', COL.flash, [0, dsMax]);
    const resG3 = drawEmptyRes(s3Svg, TOY.N, TOY.D, 10 + l3W + opGap + r3W + opGap, oY, 'ΔS');
    s3Svg.append('text').attr('x', 10 + l3W + opGap + r3W + opGap + (TOY.D * TSTP) / 2)
        .attr('y', oY + res3H + 18).attr('text-anchor', 'middle')
        .attr('fill', '#52525b').attr('font-size', '9px').text('→ slides over & updates S');

    await sleep(300);
    await animateOuterProduct(lG3, rG3, resG3, deltaS, TOY.N, TOY.D,
        v => dsScale(Math.abs(v)), 130);

    computeWrite();

    // Re-render S matrix to show update
    const sViz = document.getElementById('inf-s-viz');
    if (sViz) {
        sViz.innerHTML = '';
        const sMaxNew = Math.max(0.01, ...state.S.flat().map(Math.abs));
        const dynBNew = cscale(COL.residual.bg, COL.residual.border, [0, sMaxNew]);
        const sSvgNew = d3.select('#inf-s-viz').append('svg')
            .attr('viewBox', `0 0 ${TOY.D * TSTP + 70} ${TOY.N * TSTP + 30}`)
            .attr('width', '100%').attr('preserveAspectRatio', 'xMidYMid meet')
            .style('max-width', '240px').style('background', '#0a0a12').style('border-radius', '8px');
        drawStaticMat(sSvgNew, state.S, TOY.N, TOY.D, v => dynBNew(Math.abs(v)),
            20, 20, `S [${HP.Nh.toLocaleString()}×${HP.D}] — UPDATED`);
    }

    const activeN = state.xt.filter(v => v > 0).length;
    s3Status.textContent = `✓ ${activeN} rows updated — Hebbian write complete. S now contains memory of this token.`;
    s3Status.className = 'inf-step-status done';
}

// ══════════════════════════════════════════════════════════════════════════════
//  HYPERPARAMETER TABLE
// ══════════════════════════════════════════════════════════════════════════════

function buildHPTable() {
    return `<table class="arch-hp-table">
        <tr><th>Symbol</th><th>Name</th><th>Value</th></tr>
        <tr><td>B</td><td>Batch Size</td><td>${HP.B}</td></tr>
        <tr><td>T</td><td>Sequence Length</td><td>${HP.T.toLocaleString()} (train) / 1 (infer)</td></tr>
        <tr><td>D</td><td>Latent "Radio" Dim</td><td>${HP.D}</td></tr>
        <tr><td>L</td><td>Layers</td><td>${HP.L}</td></tr>
        <tr><td>H</td><td>Heads</td><td>${HP.H}</td></tr>
        <tr><td>N</td><td>Brain/Neuron Dim</td><td>${HP.N.toLocaleString()}</td></tr>
        <tr><td>N<sub>h</sub></td><td>Brain Dim Per Head</td><td>${HP.N.toLocaleString()} / ${HP.H} = ${HP.Nh.toLocaleString()}</td></tr>
        <tr><td>V</td><td>Vocab Size</td><td>${HP.V.toLocaleString()}</td></tr>
    </table>`;
}

// ══════════════════════════════════════════════════════════════════════════════
//  PUBLIC INTERFACE
// ══════════════════════════════════════════════════════════════════════════════

export function initInferenceViz(container) {
    container.innerHTML = '';

    const wrap = document.createElement('div');
    wrap.className = 'arch-wrap';
    wrap.innerHTML = `
        <!-- Header -->
        <div class="arch-header">
            <div class="arch-title-row">
                <h2 class="arch-title">BDH Architecture — Exhaustive Visual Explainer</h2>
                <div class="arch-mode-badge">TRAINING MODE — Parallel Tensor Flow</div>
            </div>
            <div class="arch-hp-wrap">${buildHPTable()}</div>
            <div class="arch-legend">
                <span class="arch-legend-chip" style="--chip-color:${COL.residual.border}">Latent / Residual (D=${HP.D})</span>
                <span class="arch-legend-chip" style="--chip-color:${COL.neuron.border}">Neuron / Brain (N=${HP.N.toLocaleString()})</span>
                <span class="arch-legend-chip" style="--chip-color:${COL.attn.border}">Attention Core</span>
                <span class="arch-legend-chip" style="--chip-color:${COL.weight.border}">Weight Matrices</span>
                <span class="arch-legend-chip" style="--chip-color:${COL.output.border}">Output / Readout</span>
            </div>
        </div>

        <!-- Static SVG Pipeline -->
        <div class="arch-svg-wrap" id="arch-svg-container"></div>

        <!-- Inference Popup Overlay -->
        <div id="iv-overlay" class="iv-overlay">
            <div id="iv-modal" class="iv-modal">
                <button id="iv-modal-close" class="iv-modal-close">✕</button>
                <div class="iv-modal-title" style="color:${COL.attn.border}">⚡ RECURRENT INFERENCE MODE — O(1) Memory</div>
                <div class="iv-modal-formula">
                    State-Space Mode: T=1 (sequential) — Fixed State Matrix S replaces growing KV-cache
                </div>
                <div id="iv-modal-body" class="iv-modal-body"></div>
            </div>
        </div>
    `;
    container.appendChild(wrap);

    // Render SVG
    const svgContainer = document.getElementById('arch-svg-container');
    const svg = d3.select(svgContainer).append('svg')
        .attr('id', 'arch-svg')
        .attr('width', '100%')
        .attr('preserveAspectRatio', 'xMidYMid meet')
        .style('background', COL.bg)
        .style('border-radius', '12px');

    state.svg = svg;

    const totalH = renderStaticPipeline(svg);
    svg.attr('viewBox', `0 0 ${LY.W} ${totalH}`);

    setupHighlights(svg);
}
