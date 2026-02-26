
// ──────────────────────────────────────────────────────────────────────────────
// graph.js — Force-directed network graph (professional styling)
// ──────────────────────────────────────────────────────────────────────────────

let simulation;
let svg;
let link, node;
let topologyData;
let width, height;

// Selection state
let selectedType = null;   // 'neuron' | 'edge' | null
let selectedId = null;   // neuron index OR edge key "src-tgt"

// Delta tracking — stores values from the previous updateGraph call
let prevFlows = {};        // edge key → flow value from last frame
let prevActivations = [];  // activation per neuron from last frame

// Callback set by main.js
let onSelectCallback = null;

// Visual intensity: 'low' | 'medium' | 'high'
let visualIntensity = 'low';

const INTENSITY_SCALES = {
    low: { nodeOpacity: 0.6, edgeOpacity: 0.5, haloOpacity: 0.4 },
    medium: { nodeOpacity: 0.8, edgeOpacity: 0.7, haloOpacity: 0.6 },
    high: { nodeOpacity: 1.0, edgeOpacity: 1.0, haloOpacity: 1.0 },
};

export function setIntensity(level) {
    visualIntensity = level;
}

export function setOnSelect(cb) {
    onSelectCallback = cb;
}

export function initGraph(container, topology) {
    topologyData = topology;
    width = container.clientWidth;
    height = container.clientHeight;

    // Reset delta tracking
    prevFlows = {};
    prevActivations = [];

    // ── SVG ─────────────────────────────────────────────────────────────────
    svg = d3.select(container).append('svg')
        .attr('width', '100%')
        .attr('height', '100%')
        .attr('viewBox', [0, 0, width, height])
        .style('background-color', '#09090b');

    // ── Defs (Starry Glow Filters)  ──────────────────────────────
    const defs = svg.append('defs');

    // Subtle star glow
    const starGlow = defs.append('filter').attr('id', 'star-glow').attr('x', '-50%').attr('y', '-50%').attr('width', '200%').attr('height', '200%');
    starGlow.append('feGaussianBlur').attr('stdDeviation', '2.5').attr('result', 'coloredBlur');
    const feMerge1 = starGlow.append('feMerge');
    feMerge1.append('feMergeNode').attr('in', 'coloredBlur');
    feMerge1.append('feMergeNode').attr('in', 'SourceGraphic');

    // Heavy blooming glow for highly active stars
    const heavyGlow = defs.append('filter').attr('id', 'heavy-glow').attr('x', '-50%').attr('y', '-50%').attr('width', '200%').attr('height', '200%');
    heavyGlow.append('feGaussianBlur').attr('stdDeviation', '6').attr('result', 'coloredBlur');
    const feMerge2 = heavyGlow.append('feMerge');
    feMerge2.append('feMergeNode').attr('in', 'coloredBlur');
    feMerge2.append('feMergeNode').attr('in', 'SourceGraphic');

    // ── Zoom ────────────────────────────────────────────────────────────────
    const zoom = d3.zoom()
        .scaleExtent([0.1, 8])
        .on('zoom', event => g.attr('transform', event.transform));

    svg.call(zoom);

    // Click on SVG background → deselect
    svg.on('click', () => {
        if (d3.event && d3.event.target === svg.node()) clearSelection();
    });

    // ── Main group ──────────────────────────────────────────────────────────
    const g = svg.append('g');

    // ── Simulation ──────────────────────────────────────────────────────────
    simulation = d3.forceSimulation(topology.nodes)
        .force('link', d3.forceLink(topology.edges).id(d => d.id).distance(55).strength(0.08))
        .force('charge', d3.forceManyBody().strength(-120))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('x', d3.forceX(width / 2).strength(0.08))
        .force('y', d3.forceY(height / 2).strength(0.08))
        .force('collide', d3.forceCollide(14));

    // ── Links ────────────────────────────────────────────────────────────────
    link = g.append('g').attr('class', 'links')
        .selectAll('line')
        .data(topology.edges)
        .join('line')
        .attr('class', 'edge-line')
        .attr('stroke-width', d => 0.4 + Math.sqrt(Math.abs(d.weight)) * 1.5)
        .attr('stroke', 'rgba(160, 190, 255, 0.05)') // Faint deep-space blue lines
        .attr('stroke-opacity', 0.2)
        .attr('cursor', 'pointer')
        .on('click', (event, d) => {
            event.stopPropagation();
            const key = `${d.source.index}-${d.target.index}`;
            if (selectedType === 'edge' && selectedId === key) {
                clearSelection();
            } else {
                selectEdge(d, key);
            }
        })
        .on('mouseenter', function (event, d) {
            if (selectedType === 'edge' && selectedId === `${d.source.index}-${d.target.index}`) return;
            d3.select(this)
                .attr('stroke', 'rgba(230,240,255,0.6)')
                .attr('stroke-opacity', 0.8)
                .attr('stroke-width', 2)
                .style('filter', 'url(#star-glow)');
        })
        .on('mouseleave', function (event, d) {
            if (selectedType === 'edge' && selectedId === `${d.source.index}-${d.target.index}`) return;
            d3.select(this)
                .attr('stroke', 'rgba(160, 190, 255, 0.05)')
                .attr('stroke-opacity', 0.2)
                .attr('stroke-width', 0.4)
                .style('filter', null);
        });


    // ── Nodes ────────────────────────────────────────────────────────────────
    node = g.append('g').attr('class', 'nodes')
        .selectAll('.node')
        .data(topology.nodes)
        .join('g')
        .attr('class', 'node')
        .attr('cursor', 'pointer')
        .call(drag(simulation))
        .on('click', (event, d) => {
            event.stopPropagation();
            const idx = d.index;
            if (selectedType === 'neuron' && selectedId === idx) {
                clearSelection();
            } else {
                selectNeuron(d, idx);
            }
        });

    // Outer aura (prev activation / blooming effect)
    node.append('circle')
        .attr('class', 'ring')
        .attr('r', 8)
        .attr('fill', 'rgba(160, 190, 255, 0.02)')
        .attr('stroke', 'rgba(160, 190, 255, 0.1)')
        .attr('stroke-width', 0.5);

    // Inner core (the star itself)
    node.append('circle')
        .attr('class', 'core')
        .attr('r', 2)
        .attr('fill', 'rgba(120, 140, 160, 0.4)')
        .attr('stroke', 'none')
        .style('filter', 'url(#star-glow)');

    // Selection halo — flat amber ring, no glow filter
    node.append('circle')
        .attr('class', 'sel-halo')
        .attr('r', 12)
        .attr('fill', 'none')
        .attr('stroke', '#0ea5e9')
        .attr('stroke-width', 2)
        .attr('opacity', 0);

    // Tooltip
    node.append('title')
        .text(d => `Neuron ${d.original_idx ?? d.id}  (click to track)`);

    // ── Tick ─────────────────────────────────────────────────────────────────
    simulation.on('tick', () => {
        link.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
        node.attr('transform', d => `translate(${d.x},${d.y})`);
    });
}

// ── update (called every frame) ────────────────────────────────────────────────
export function updateGraph(frame) {
    if (!node || !link) return;

    const { activations, prev_activations } = frame;
    const iScale = INTENSITY_SCALES[visualIntensity] || INTENSITY_SCALES.low;

    // ── Nodes ─────────────────────────────────────────────────────────────────
    node.each(function (d, i) {
        const grp = d3.select(this);
        const xVal = activations[i];
        const prev = prevActivations.length > i ? prevActivations[i] : xVal;
        const delta = xVal - prev;

        // Size: stars are tiny when faint, bloom larger when active
        const t = Math.min(xVal / 2.0, 1.0);
        const radius = 2 + t * 4;

        // Core: from faint deep blue-gray to bright glowing white/yellow
        const coreBrightness = Math.min(xVal / 1.5, 1.0);
        const coreColor = d3.interpolateRgb('rgba(100,120,150,0.3)', 'rgba(255,250,230,1.0)')(coreBrightness);
        grp.select('.core')
            .attr('fill', coreColor)
            .attr('r', radius)
            .attr('stroke', 'none')
            .style('filter', coreBrightness > 0.6 ? 'url(#heavy-glow)' : 'url(#star-glow)');

        // Outer aura: expands into a gentle nebula glow when star's activation changes significantly
        const THRESHOLD = 0.005;
        let ringRatio = 0;
        if (Math.abs(delta) > THRESHOLD) {
            ringRatio = Math.min(Math.abs(delta) * 8, 1.0);
        }

        let ringRadius = radius + 2 + ringRatio * 6;
        let ringColor = d3.interpolateRgb('rgba(160,190,255,0.01)', 'rgba(200,230,255,0.4)')(ringRatio * iScale.nodeOpacity);
        let ringWidth = 0.5 + ringRatio * 2;

        grp.select('.ring')
            .attr('r', ringRadius)
            .attr('stroke', ringColor)
            .attr('fill', ringRatio > 0.2 ? 'rgba(160,190,255,0.05)' : 'none')
            .attr('stroke-width', ringWidth)
            .style('filter', 'url(#star-glow)');

    });

    // ── Edges ─────────────────────────────────────────────────────────────────
    const newFlows = {};
    link.each(function (d) {
        const el = d3.select(this);
        const key = `${d.source.index}-${d.target.index}`;
        if (selectedType === 'edge' && selectedId === key) {
            newFlows[key] = prevFlows[key] ?? 0;
            return; // keep selection highlight
        }

        const ySrc = prev_activations[d.source.index];
        const xTgt = activations[d.target.index];
        const flow = ySrc * d.weight * xTgt;
        const flowAbs = Math.abs(flow);
        const flowT = Math.min(flowAbs * 12.0, 1.0);

        // Delta vs previous frame
        const prevFlow = prevFlows[key] ?? flow;
        const flowDelta = flow - prevFlow;
        newFlows[key] = flow;

        const EDGE_THRESH = 0.0005;
        let strokeColor, strokeOpacity, strokeWidth;

        if (flowT < 0.03) {
            strokeColor = 'rgba(160, 190, 255, 0.05)';
            strokeOpacity = 0.1 * iScale.edgeOpacity;
            strokeWidth = 0.4;
            el.style('filter', null);
        } else {
            // Flow active — light up the constellation line
            const brightness = Math.min(flowAbs * 15, 1.0);
            strokeColor = d3.interpolateRgb('rgba(160,190,255,0.1)', 'rgba(230,245,255,0.9)')(brightness);
            strokeOpacity = (0.2 + brightness * 0.8) * iScale.edgeOpacity;
            strokeWidth = 0.5 + brightness * 2.5;
            el.style('filter', brightness > 0.5 ? 'url(#star-glow)' : null);
        }

        el.attr('stroke', strokeColor)
            .attr('stroke-opacity', strokeOpacity)
            .attr('stroke-width', strokeWidth);
    });

    prevFlows = newFlows;
    prevActivations = Array.from(activations);

    // Keep selection halos correct
    if (selectedType === 'neuron') {
        node.select('.sel-halo').attr('opacity', d => d.index === selectedId ? 0.8 : 0);
    }
    if (selectedType === 'edge') {
        link.each(function (d) {
            const key = `${d.source.index}-${d.target.index}`;
            if (key === selectedId) {
                d3.select(this)
                    .attr('stroke', '#0ea5e9')
                    .attr('stroke-opacity', 0.85)
                    .attr('stroke-width', 2.5);
            }
        });
    }
}

export function cleanupGraph() {
    if (simulation) simulation.stop();
}

// ── Selection helpers ──────────────────────────────────────────────────────────
function clearSelection() {
    selectedType = null;
    selectedId = null;

    if (node) node.select('.sel-halo').attr('opacity', 0);
    if (onSelectCallback) onSelectCallback(null);
}

function selectNeuron(d, idx) {
    selectedType = 'neuron';
    selectedId = idx;

    node.select('.sel-halo').attr('opacity', nd => nd.index === idx ? 0.8 : 0);
    // Restore any highlighted edge
    link.attr('filter', null);

    if (onSelectCallback) {
        onSelectCallback({ type: 'neuron', index: idx, label: `Neuron ${d.original_idx ?? d.id}` });
    }
}

function selectEdge(d, key) {
    selectedType = 'edge';
    selectedId = key;

    // Clear neuron halo
    if (node) node.select('.sel-halo').attr('opacity', 0);

    if (onSelectCallback) {
        const srcLabel = `${d.source.original_idx ?? d.source.index}`;
        const tgtLabel = `${d.target.original_idx ?? d.target.index}`;
        onSelectCallback({ type: 'edge', srcIdx: d.source.index, tgtIdx: d.target.index, weight: d.weight, label: `Edge ${srcLabel} → ${tgtLabel}  (w=${d.weight.toFixed(3)})` });
    }
}

// ── Drag ──────────────────────────────────────────────────────────────────────
function drag(sim) {
    return d3.drag()
        .on('start', (event) => {
            if (!event.active) sim.alphaTarget(0.3).restart();
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
        })
        .on('drag', (event) => {
            event.subject.fx = event.x;
            event.subject.fy = event.y;
        })
        .on('end', (event) => {
            if (!event.active) sim.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
        });
}
