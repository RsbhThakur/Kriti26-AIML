# BDH Visualizer — KRITI Path A Final Submission

Animation-first visual analytics for Brain-Derived Heuristics (BDH).

Repository: `[Link Text](https://github.com/RsbhThakur/Kriti26-AIML)`  
Live Demo: `[Link Text](https://rsbhthakur.github.io/Kriti26-AIML/web/setup.html)`  
Youtube Demo: `[Link Text](https://youtu.be/-gRUDOUw85g?si=UiLSYuq1dQY-pfPJ)`  

---

## Project Structure

All code lives in `web/`:

```
web/
├── index.html          # main visualization (11 tabs)
├── setup.html          # start/end coordinate picker
├── css/
│   └── style.css
├── js/
│   ├── main.js         # playback, telemetry, tab switching
│   ├── advanced_panels.js  # attention, concept probe, scaling lab
│   ├── graph.js        # force-directed neural graph
│   ├── panels.js       # sparse brain, topology, memory
│   ├── hebbian.js      # Hebbian weight animation
│   ├── viz3d.js        # Three.js 3D walkthrough
│   ├── chart.js        # activation history chart
│   ├── board.js        # pathfinder board
│   └── inference_viz.js
└── data/
    ├── viz_data.json       # pre-computed inference output (~40MB)
    └── hebbian_data.json   # pre-computed Hebbian data (~2MB)
```

---

## Visualization Tabs

| Tab | What it shows |
|---|---|
| Graph | Force-directed neural activation graph |
| Live Analysis | Per-frame telemetry, top-20 neurons, activation histogram |
| 3D Walkthrough | Three.js 3D neural field with orbit controls |
| Sparse Brain | 16×16 activation heatmap with path overlay |
| Topology | Graph community detection and hub analysis |
| Attention Atlas | Layer-wise attention heatmap with animation |
| Concept Probe | Functional phase assignment per neuron |
| Hebbian | Multi-board weight evolution animation |
| Memory Formation | Dual-stream x/y activation history charts |
| Scaling Lab | O(T) vs O(T²) interactive comparison |
| Inference | Step-by-step BDH compute cycle replay |

---

## Run Locally

```bash
cd bdh-visualizer/web
python -m http.server 8080
```

Open: `http://localhost:8080/setup.html`

---

## Deploy

### GitHub Pages

```bash
cd bdh-visualizer/web
git init
git add .
git commit -m "BDH static build"
git remote add origin https://github.com/<user>/<repo>.git
git push -u origin main
```

GitHub repo → Settings → Pages → Branch: `main`, Folder: `/` → Save.

### Netlify

Drag the `web/` folder onto [netlify.com/drop](https://app.netlify.com/drop).

---

## Submission Checklist

- [ ] Hosted demo link (public and working)
- [ ] Public GitHub repository
- [ ] 2–3 minute demo video
- [ ] README with: what was built, insights shown, local run steps, hosted link, team contributions, limitations

---

## References

- BDH paper: `https://arxiv.org/abs/2509.26507`
- Transformer Explainer: `https://poloclub.github.io/transformer-explainer/`
- Pathway: `https://pathway.com/`
