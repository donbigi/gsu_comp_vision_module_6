# CSc 8830 · Assignment 6 — Optical Flow & Structure from Motion

Synthetic-data implementation of the two parts of the assignment:

1. **Optical flow & motion tracking** on two 30 s videos with known motion,
   visualised as video and validated against exact ground truth.
2. **Structure from motion** of a planar object from four calibrated viewpoints,
   reconstructed by homography DLT and by multi-view triangulation.

Everything is generated synthetically so that every "measured" quantity has an
exact, known ground truth — no hand-labelling, no calibration error.

## Files

| File | Purpose |
|------|---------|
| `synthetic_data.py` | Generates the two 30 s videos + the four SfM viewpoints and camera data. Exposes the analytic motion models. |
| `optical_flow.py` | Dense (Farnebäck) + sparse (Lucas–Kanade) flow, colour-wheel video, and tracking validation. |
| `structure_from_motion.py` | Homography DLT, planar reconstruction, 4-view triangulation, boundary comparison. |
| `generate_results.py` | Assembles `static/` (figures, flow animations, `results.json`) from `data/` for the web demo. |
| `app.py` | Flask dashboard that serves the precomputed results (no compute in the pod). |
| `DERIVATIONS.md` | All hand-derived math (tracking equations, bilinear interpolation, SfM) — convert to PDF with pandoc. |
| `data/` | Generated outputs (videos, figures, `sfm_data.json`, `validation.json`). |

## Setup

```bash
pip install -r requirements-dev.txt
```

Requires Python ≥ 3.9 with `numpy`, `opencv-python`, `matplotlib`, `pillow`
(tested on Python 3.11 / OpenCV 5.0).  `requirements.txt` is the *web app's*
dependency set (Flask only); the pipeline itself uses `requirements-dev.txt`.

## Run

Run the three scripts in order:

```bash
python synthetic_data.py          # ~30 s: generates data/ (videos + SfM views)
python optical_flow.py            # computes + validates flow, writes flow videos
python structure_from_motion.py   # reconstructs the planar object, writes figure
```

### What each produces (in `data/`)

- `video_1.mp4` — 30 s @ 30 fps, horizontal pan + orbiting textured disk.
- `video_2.mp4` — 30 s @ 30 fps, dolly zoom in/out + oscillating textured square.
- `flow_viz_1.mp4`, `flow_viz_2.mp4` — optical-flow visualisation videos
  (Middlebury colour wheel: hue = direction, value = magnitude).
- `flow_pair_1.png`, `flow_pair_2.png` — two consecutive frames + dense flow +
  LK-tracked vs. ground-truth arrows.
- `sfm_view_1..4.png` — the planar object from four cameras.
- `sfm_data.json` — camera intrinsics/poses + ground-truth 3-D points + projections.
- `sfm_reconstruction.png` — the four views + true vs. reconstructed boundary.
- `validation.json` — numeric validation results.

## Results

### Optical flow — tracking validation (frames t = 150, 151)

400 Shi–Tomasi corners tracked with pyramidal Lucas–Kanade, compared to the
analytic ground-truth displacement:

| video | motion | LK mean error | LK median | Farnebäck MEE |
|-------|--------|---------------|-----------|---------------|
| 1 | pan (3 px/frame) + orbiting disk | **0.083 px** | 0.001 px | 0.025 px |
| 2 | dolly zoom + oscillating square | **0.122 px** | 0.091 px | 0.115 px |

Worked example (video 1, background corner): theoretical $(-3.00, 0.00)$ px,
LK tracked $(-3.0001, 0.0001)$ px → error $2\times10^{-4}$ px.
Worked example (video 2, background corner): theoretical $(-2.27, -2.48)$ px,
LK tracked it within $0.13$ px.

**What the flow reveals** (evidence in `flow_viz_*.mp4` / `flow_pair_*.png`):

- *Video 1*: the colour wheel is almost a single hue (uniform leftward motion)
  → a **camera pan**, with a small localised region of different hue moving
  opposite to the background → an **independently moving object** (motion
  segmentation).
- *Video 2*: the hue forms a **radial pattern** centred on the image centre →
  forward/backward motion with a **focus of expansion / contraction**; the
  localised different-coloured region is the oscillating square.

### Structure from motion

Four views of a planar object (world $Z=0$), reconstructed two ways:

- Homography DLT vs. $K[\mathbf r_1\ \mathbf r_2\ t]$: relative error **0** (all views).
- Planar reconstruction mean error: **0.0 mm**.
- 4-view triangulation mean error: **0.0 mm**, recovered $Z \approx 0$.
- Boundary area recovered exactly ($4400\ \mathrm{mm}^2$).

Camera intrinsics and the four camera positions are in §B.6 of
`DERIVATIONS.md`, printed by `structure_from_motion.py`, and stored in
`data/sfm_data.json`.

## Making the report

```bash
pandoc DERIVATIONS.md -o DERIVATIONS.pdf
```

The PDF report should include: the DERIVATIONS material, the result figures
(`flow_pair_*.png`, `sfm_reconstruction.png`), and a link to this GitHub repo.
Record a screen capture of `optical_flow.py` + `structure_from_motion.py`
running (or play `flow_viz_1.mp4` / `flow_viz_2.mp4`) for the video submission.

## Web demo (Docker)

A static Flask dashboard serves the precomputed results (the same flow/SfM
figures and metrics documented above) — the heavy pipeline runs offline, never
in the pod.

```bash
python generate_results.py                 # assemble static/ from data/
docker build -t gsu-cv-module6-flow .      # or: docker compose up
docker run -p 8000:8000 gsu-cv-module6-flow
# open http://localhost:8000
```

Files: `app.py` (Flask, serves `static/results.json`), `templates/index.html`,
`static/app.js` / `static/style.css`, `Dockerfile`, `docker-compose.yml`,
`module6.yaml` (Kubernetes + Istio, host `module6.ucosibe.xyz`).

## References

See the References section of `DERIVATIONS.md` (Horn & Schunck 1981; Lucas &
Kanade 1981; Farnebäck 2003; Shi & Tomasi 1994; Bouguet 2000; Hartley &
Zisserman 2004; Szeliski 2010; OpenCV docs).
