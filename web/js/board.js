
// Board Visualization Logic — Professional styling (no glows)

let boardContainer;
let cells = [];
let inputBoard = null;

// Desaturated, professional colors
const CELL_COLORS = {
    0: '#18181b', // FLOOR (zinc-900)
    1: '#334155', // WALL (slate-700)
    2: '#10b981', // START (emerald)
    3: '#f43f5e', // END (rose)
    4: '#f59e0b', // PATH (amber)
    5: '#8b5cf6', // WAYPOINT (violet)
    6: '#f59e0b', // PATH (amber)
    7: '#18181b'  // SEARCHED
};

export function initBoard(container, size, fixedInputBoard) {
    boardContainer = container;
    inputBoard = fixedInputBoard || null;

    container.innerHTML = '';

    const wrapper = document.createElement('div');
    wrapper.className = 'board-grid';
    wrapper.style.gridTemplateColumns = `repeat(${size}, 1fr)`;
    wrapper.style.gridTemplateRows = `repeat(${size}, 1fr)`;

    cells = [];

    for (let i = 0; i < size * size; i++) {
        const cell = document.createElement('div');
        cell.className = 'board-cell';

        const row = Math.floor(i / size);
        const col = i % size;
        cell.dataset.idx = i;
        cell.title = `(${row}, ${col})`;

        wrapper.appendChild(cell);
        cells.push(cell);
    }

    container.appendChild(wrapper);
}

export function updateBoard(frame, size) {
    const predictions = frame.board_prediction;

    for (let i = 0; i < cells.length; i++) {
        const cell = cells[i];
        let val = predictions[i];

        // ALWAYS overlay fixed input board markers
        if (inputBoard && (inputBoard[i] === 2 || inputBoard[i] === 3)) {
            val = inputBoard[i];
        }

        const color = CELL_COLORS[val] || CELL_COLORS[0];
        cell.style.backgroundColor = color;
        // No glow — flat clean tiles
        cell.style.boxShadow = 'none';
    }

    // Draw Attention Arcs (if present)
    const oldSvg = boardContainer.querySelector('svg.attn-layer');
    if (oldSvg) oldSvg.remove();

    if (frame.attention && frame.attention.length > 0) {
        const attn = frame.attention;
        const T = attn.length;

        const wrapper = boardContainer.querySelector('div');
        const rect = wrapper.getBoundingClientRect();

        const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        svg.setAttribute("class", "attn-layer");
        svg.style.position = "absolute";
        svg.style.top = "0";
        svg.style.left = "0";
        svg.style.width = "100%";
        svg.style.height = "100%";
        svg.style.pointerEvents = "none";

        const cellW = rect.width / size;
        const cellH = rect.height / size;

        const flatAttn = [];
        for (let i = 0; i < T; i++) {
            for (let j = 0; j < T; j++) {
                if (i !== j) {
                    flatAttn.push({ src: i, tgt: j, val: attn[i][j] });
                }
            }
        }
        flatAttn.sort((a, b) => b.val - a.val);
        const topK = flatAttn.slice(0, 20);

        topK.forEach(edge => {
            const { src, tgt, val } = edge;
            if (val < 0.005) return;

            const srcR = Math.floor(src / size);
            const srcC = src % size;
            const tgtR = Math.floor(tgt / size);
            const tgtC = tgt % size;

            const x1 = (srcC + 0.5) * cellW;
            const y1 = (srcR + 0.5) * cellH;
            const x2 = (tgtC + 0.5) * cellW;
            const y2 = (tgtR + 0.5) * cellH;

            const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
            line.setAttribute("x1", x1);
            line.setAttribute("y1", y1);
            line.setAttribute("x2", x2);
            line.setAttribute("y2", y2);
            line.setAttribute("stroke", "rgba(79,142,252,0.4)");
            line.setAttribute("stroke-width", Math.max(0.5, val * 2.5));
            line.setAttribute("stroke-opacity", Math.min(0.5, val * 4));

            const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
            dot.setAttribute("cx", x2);
            dot.setAttribute("cy", y2);
            dot.setAttribute("r", 1.5);
            dot.setAttribute("fill", "rgba(79,142,252,0.5)");
            dot.setAttribute("fill-opacity", Math.min(0.6, val * 4));

            svg.appendChild(line);
            svg.appendChild(dot);
        });

        boardContainer.appendChild(svg);
    }
}
