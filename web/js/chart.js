
// ────────────────────────────────────────────────────────────────
// chart.js — Activation History Chart (D3 v7) — Professional styling
// ────────────────────────────────────────────────────────────────

let chartSvg = null;
let chartContainer = null;

const MARGIN = { top: 26, right: 20, bottom: 36, left: 48 };

export function initChart(container) {
    chartContainer = container;
}

/**
 * Draw (or redraw) the chart.
 * @param {Array<{layer:number, value:number, stepName:string, stepIndex:number}>} series
 * @param {string} label
 * @param {number|null} currentLayer
 */
export function drawChart(series, label, currentLayer) {
    if (!chartContainer) return;

    chartContainer.innerHTML = '';

    const W = chartContainer.clientWidth || 520;
    const H = chartContainer.clientHeight || 220;
    const w = W - MARGIN.left - MARGIN.right;
    const h = H - MARGIN.top - MARGIN.bottom;

    const svg = d3.select(chartContainer)
        .append('svg')
        .attr('width', W)
        .attr('height', H)
        .style('overflow', 'visible');

    chartSvg = svg;

    const g = svg.append('g')
        .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    // ── Scales ────────────────────────────────────────────────────
    const layers = series.map(d => d.layer);
    const minL = d3.min(layers);
    const maxL = d3.max(layers);

    const xScale = d3.scaleLinear()
        .domain([minL, maxL])
        .range([0, w]);

    const vals = series.map(d => d.value);
    const minV = d3.min(vals);
    const maxV = d3.max(vals);
    const pad = (maxV - minV) * 0.12 || 0.1;

    const yScale = d3.scaleLinear()
        .domain([Math.min(0, minV - pad), maxV + pad])
        .range([h, 0])
        .nice();

    // ── Background grid — subtle dotted ─────────────────────────
    g.append('g')
        .attr('class', 'chart-grid')
        .call(
            d3.axisLeft(yScale)
                .ticks(5)
                .tickSize(-w)
                .tickFormat('')
        )
        .call(gg => {
            gg.select('.domain').remove();
            gg.selectAll('line')
                .attr('stroke', 'rgba(255,255,255,0.03)')
                .attr('stroke-dasharray', '2,4');
        });

    // Zero line
    if (yScale.domain()[0] < 0) {
        g.append('line')
            .attr('x1', 0).attr('x2', w)
            .attr('y1', yScale(0)).attr('y2', yScale(0))
            .attr('stroke', 'rgba(145,160,173,0.12)')
            .attr('stroke-dasharray', '4,2')
            .attr('stroke-width', 1);
    }

    // ── Step region shading — very subtle ────────────────────────
    const STEP_COLORS = ['#6366f1', '#8b5cf6', '#14b8a6', '#0ea5e9'];
    const STEP_NAMES = ['Recall', 'Mechanism', 'Effect', 'Update'];

    const byLayer = d3.group(series, d => d.layer);
    const layerArr = Array.from(byLayer.entries()).sort((a, b) => a[0] - b[0]);

    for (let i = 0; i < layerArr.length - 1; i++) {
        const [ly, pts] = layerArr[i];
        const [ly2] = layerArr[i + 1];
        const sIdx = pts[0].stepIndex;
        g.append('rect')
            .attr('x', xScale(ly))
            .attr('y', 0)
            .attr('width', xScale(ly2) - xScale(ly))
            .attr('height', h)
            .attr('fill', STEP_COLORS[sIdx % 4])
            .attr('opacity', 0.03);
    }

    // ── Area — low alpha single tone ──────────────────────────────
    const area = d3.area()
        .x(d => xScale(d.layer))
        .y0(yScale(Math.max(0, yScale.domain()[0])))
        .y1(d => yScale(d.value))
        .curve(d3.curveCatmullRom.alpha(0.5));

    const gradId = `area-grad-${Date.now()}`;
    const defs = svg.append('defs');
    const grad = defs.append('linearGradient')
        .attr('id', gradId)
        .attr('x1', '0%').attr('y1', '0%')
        .attr('x2', '0%').attr('y2', '100%');
    grad.append('stop').attr('offset', '0%')
        .attr('stop-color', '#6366f1').attr('stop-opacity', 0.10);
    grad.append('stop').attr('offset', '100%')
        .attr('stop-color', '#6366f1').attr('stop-opacity', 0.01);

    g.append('path')
        .datum(series)
        .attr('fill', `url(#${gradId})`)
        .attr('d', area);

    // ── Line — 2.5px accent-primary ──────────────────────────────
    const line = d3.line()
        .x(d => xScale(d.layer))
        .y(d => yScale(d.value))
        .curve(d3.curveCatmullRom.alpha(0.5));

    g.append('path')
        .datum(series)
        .attr('fill', 'none')
        .attr('stroke', '#6366f1')
        .attr('stroke-width', 2.5)
        .attr('stroke-linecap', 'round')
        .attr('stroke-linejoin', 'round')
        .attr('d', line);

    // ── Dots — small, color-coded by step ────────────────────────
    const dotG = g.append('g');
    dotG.selectAll('circle')
        .data(series)
        .join('circle')
        .attr('cx', d => xScale(d.layer))
        .attr('cy', d => yScale(d.value))
        .attr('r', 3)
        .attr('fill', d => STEP_COLORS[d.stepIndex % 4])
        .attr('stroke', '#09090b')
        .attr('stroke-width', 1.5);

    // ── Axes — muted ──────────────────────────────────────────────
    const xAxis = d3.axisBottom(xScale)
        .ticks(Math.min(maxL - minL + 1, 20))
        .tickFormat(d3.format('d'));

    g.append('g')
        .attr('transform', `translate(0,${h})`)
        .call(xAxis)
        .call(ag => {
            ag.select('.domain').attr('stroke', 'rgba(255,255,255,0.04)');
            ag.selectAll('text')
                .attr('fill', '#a1a1aa')
                .attr('font-size', '10px')
                .attr('font-family', "'IBM Plex Mono', monospace");
            ag.selectAll('line').attr('stroke', 'rgba(255,255,255,0.04)');
        });

    g.append('text')
        .attr('x', w / 2)
        .attr('y', h + 30)
        .attr('fill', '#71717a')
        .attr('font-size', '10px')
        .attr('text-anchor', 'middle')
        .text('Layer');

    const yAxis = d3.axisLeft(yScale).ticks(5).tickFormat(d => d.toFixed(2));
    g.append('g')
        .call(yAxis)
        .call(ag => {
            ag.select('.domain').attr('stroke', 'rgba(255,255,255,0.04)');
            ag.selectAll('text')
                .attr('fill', '#a1a1aa')
                .attr('font-size', '10px')
                .attr('font-family', "'IBM Plex Mono', monospace");
            ag.selectAll('line').attr('stroke', 'rgba(255,255,255,0.04)');
        });

    // ── Current-frame cursor — amber flat ──────────────────────────
    const cursorG = g.append('g').attr('class', 'cursor-group');

    const cursorLine = cursorG.append('line')
        .attr('y1', 0).attr('y2', h)
        .attr('stroke', '#0ea5e9')
        .attr('stroke-width', 1.5)
        .attr('stroke-dasharray', '4,2')
        .attr('opacity', 0);

    const cursorDot = cursorG.append('circle')
        .attr('r', 4)
        .attr('fill', '#0ea5e9')
        .attr('stroke', '#09090b')
        .attr('stroke-width', 2)
        .attr('opacity', 0);

    const cursorLabel = cursorG.append('text')
        .attr('fill', '#0ea5e9')
        .attr('font-size', '10px')
        .attr('font-family', "'IBM Plex Mono', monospace")
        .attr('y', -8)
        .attr('opacity', 0);

    // ── Interactive overlay — tooltip ─────────────────────────────
    const tooltipG = g.append('g').attr('class', 'tooltip-group').attr('opacity', 0);
    const toolTipBgColor = 'rgba(24,24,27,0.92)'; // zinc-900 with alpha
    const tipRect = tooltipG.append('rect')
        .attr('rx', 8).attr('ry', 8)
        .attr('fill', toolTipBgColor)
        .attr('stroke', 'rgba(255,255,255,0.05)')
        .attr('stroke-width', 1)
        .style('filter', 'drop-shadow(0 2px 6px rgba(0,0,0,0.5))');
    const tipText1 = tooltipG.append('text')
        .attr('fill', '#f4f4f5')
        .attr('font-size', '11px')
        .attr('font-weight', '500')
        .attr('font-family', "'IBM Plex Mono', monospace")
        .attr('x', 8).attr('y', 16);
    const tipText2 = tooltipG.append('text')
        .attr('fill', '#a1a1aa')
        .attr('font-size', '10px')
        .attr('font-family', "'IBM Plex Mono', monospace")
        .attr('x', 8).attr('y', 30);

    const bisect = d3.bisector(d => d.layer).left;
    g.append('rect')
        .attr('width', w).attr('height', h)
        .attr('fill', 'transparent')
        .on('mousemove', function (event) {
            const [mx] = d3.pointer(event);
            const lv = xScale.invert(mx);
            const idx = bisect(series, lv, 1);
            const a = series[idx - 1];
            const b = series[idx];
            const d = !b || Math.abs(lv - a.layer) < Math.abs(lv - b.layer) ? a : b;

            const cx = xScale(d.layer);
            const cy = yScale(d.value);

            const txt1 = `Layer ${d.layer}  val: ${d.value.toFixed(4)}`;
            const txt2 = d.stepName;

            tipText1.text(txt1);
            tipText2.text(txt2);

            const tw = Math.max(txt1.length, txt2.length) * 6.5 + 16;
            const th = 42;
            let tx = cx + 10;
            if (tx + tw > w) tx = cx - tw - 10;

            tipRect.attr('x', tx - 8).attr('y', cy - 38).attr('width', tw).attr('height', th);
            tipText1.attr('x', tx).attr('y', cy - 26);
            tipText2.attr('x', tx).attr('y', cy - 12);

            tooltipG.attr('opacity', 1);
        })
        .on('mouseleave', () => {
            tooltipG.attr('opacity', 0);
        });

    // ── Title ──────────────────────────────────────────────────────
    svg.append('text')
        .attr('x', MARGIN.left)
        .attr('y', 14)
        .attr('fill', '#f4f4f5')
        .attr('font-size', '12px')
        .attr('font-weight', '600')
        .attr('font-family', "'Inter', sans-serif")
        .text(label);

    // Store references for cursor updates
    chartContainer._d3State = { xScale, yScale, series, cursorLine, cursorDot, cursorLabel };

    if (currentLayer !== null) {
        updateCursor(currentLayer);
    }
}

/**
 * Update the vertical cursor line to the given layer number.
 */
export function updateCursor(layer) {
    if (!chartContainer || !chartContainer._d3State) return;
    const { xScale, yScale, series, cursorLine, cursorDot, cursorLabel } = chartContainer._d3State;

    const bisect = d3.bisector(d => d.layer).left;
    const lv = layer;
    const idx = bisect(series, lv, 1);
    const a = series[idx - 1];
    const b = series[idx];
    const d = !a ? b : (!b || Math.abs(lv - a.layer) < Math.abs(lv - b.layer) ? a : b);

    if (!d) return;

    const cx = xScale(d.layer);
    const cy = yScale(d.value);

    cursorLine
        .attr('x1', cx).attr('x2', cx)
        .attr('opacity', 1);

    cursorDot
        .attr('cx', cx).attr('cy', cy)
        .attr('opacity', 1);

    cursorLabel
        .attr('x', cx + 4)
        .attr('opacity', 1)
        .text(`${d.value.toFixed(3)}`);
}
