# BDH Visualizer (KRITI Path A Final Submission)

Submission-ready, animation-first visual analytics for Dragon Hatchling (BDH), built to match Path A judging criteria:
- visual clarity
- technical correctness
- architectural insight
- presentation quality

Repository: `https://github.com/RsbhThakur/Kriti26-AIML`

## Final Entry Point

Use `pages/5_🚀_Frontier_Studio.py` as the main demo page.  
It is the superset implementation merging the strongest parts of `version2/web` and `bdh-visualizer`:
- setup-style controls (fixed start/end, neuron budget, thresholds)
- live pathfinder board replay with animated top attention connections
- animated 2D neuron circuit and animated 3D neuron field
- multi-board Hebbian progression animation
- architecture equations and explainability narrative

The other pages remain as focused explainers:
- `1_🐉_BDH_Explainer.py`
- `2_🧠_Sparse_Brain.py`
- `3_🕸️_Graph_Brain.py`
- `4_💡_Memory_Formation.py`

## Model Resolution

Model checkpoint is resolved in this order:
1. `BDH_MODEL_PATH` environment variable
2. `../version2/boardpath.pt` (default, updated model)
3. `./model/boardpath.pt` (fallback)

Implemented in `core/runtime.py`.

## Local Run

```bash
cd bdh-visualizer
python -m venv venv
source venv/bin/activate   # Windows PowerShell: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Optional explicit model path:

```bash
export BDH_MODEL_PATH=/absolute/path/to/boardpath.pt
streamlit run app.py
```

Sanity test:

```bash
python check_setup.py
```

## Deployment

### Option A: Hugging Face Spaces (recommended for judging)
1. Create a new Space with SDK `Streamlit`.
2. Upload `bdh-visualizer` contents as the Space repository root.
3. Ensure `requirements.txt` is present.
4. Add Space variable `BDH_MODEL_PATH` if using external checkpoint path; otherwise include `model/boardpath.pt` or keep `version2/boardpath.pt` in repo structure.
5. Set app file to `app.py` (default for Streamlit Spaces).

### Option B: Streamlit Community Cloud
1. Push `bdh-visualizer` to GitHub.
2. Create new Streamlit app linked to `app.py`.
3. Add `BDH_MODEL_PATH` in app secrets/environment if needed.
4. Deploy and verify all pages, especially `Frontier Studio`.

## Reproducibility Assets

Generate JSON assets compatible with `version2/web`:

```bash
python scripts/build_submission_assets.py
```

Outputs:
- `version2/web/data/viz_data.json`
- `version2/web/data/hebbian_data.json`

## Submission Checklist

- hosted demo link (public and working)
- public GitHub repository
- 2-3 minute demo video
- README with:
  - what was built
  - insights shown about BDH
  - local run steps
  - hosted demo link
  - team contributions
  - limitations and future scope

## References

- BDH paper: `https://arxiv.org/abs/2509.26507`
- Transformer Explainer inspiration: `https://poloclub.github.io/transformer-explainer/`
- Pathway: `https://pathway.com/`
