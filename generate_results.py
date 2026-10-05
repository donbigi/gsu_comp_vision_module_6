"""
generate_results.py — assemble the Module 6 web-demo assets from data/.
=======================================================================

The three pipeline scripts write their outputs into ``data/``:

    synthetic_data.py        -> videos + SfM viewpoints + sfm_data.json
    optical_flow.py          -> flow videos + figures + validation.json
    structure_from_motion.py -> sfm_reconstruction.png

This script collects those outputs into ``static/`` (figures, downloadable flow
videos, small looping GIF previews of the flow) and builds ``static/results.json``
so the Flask app can serve everything with no heavy compute in the pod.

Run from the module directory:

    python generate_results.py              # assemble from an existing data/
    python generate_results.py --regenerate # re-run the whole pipeline first

The full pipeline needs numpy / opencv-python / matplotlib (see requirements-dev.txt);
the deployed app itself only needs Flask (requirements.txt).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import structure_from_motion as sfm

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
STATIC = BASE / "static"

GIF_N_FRAMES = 36          # looping preview frames
GIF_WIDTH = 480            # down-sampled preview width (px)
GIF_DURATION = 80          # ms per frame (~12.5 fps preview)

MOTIONS = {
    "1": "Horizontal camera pan (3 px/frame) + orbiting textured disk",
    "2": "Dolly zoom (radial flow) + oscillating textured square",
}

FIGURES = ("flow_pair_1.png", "flow_pair_2.png", "sfm_reconstruction.png",
           "sfm_view_1.png", "sfm_view_2.png", "sfm_view_3.png", "sfm_view_4.png")
VIDEOS = ("flow_viz_1.mp4", "flow_viz_2.mp4")


def run_pipeline() -> None:
    for script in ("synthetic_data.py", "optical_flow.py", "structure_from_motion.py"):
        print(f"running {script} ...")
        subprocess.run([sys.executable, str(BASE / script)], check=True)


def copy_assets() -> None:
    STATIC.mkdir(exist_ok=True)
    for name in FIGURES + VIDEOS:
        src = DATA / name
        if src.exists():
            shutil.copy2(src, STATIC / name)
        else:
            print(f"warning: {name} missing from data/; run the pipeline first")


def make_gif(video_id: int, out_name: str) -> None:
    """Down-sample the colour-wheel flow video to a small looping GIF preview."""
    src = DATA / f"flow_viz_{video_id}.mp4"
    cap = cv2.VideoCapture(str(src))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release()
        print(f"warning: could not read {src}")
        return
    idxs = np.linspace(0, n - 1, GIF_N_FRAMES).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        scale = GIF_WIDTH / w
        frame = cv2.resize(frame, (GIF_WIDTH, int(round(h * scale))))
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    if frames:
        out = STATIC / out_name
        frames[0].save(out, save_all=True, append_images=frames[1:],
                       duration=GIF_DURATION, loop=0, optimize=True)
        print(f"wrote {out} ({len(frames)} frames, {out.stat().st_size // 1024} KB)")


def optical_flow_summary() -> list[dict]:
    with open(DATA / "validation.json") as f:
        val = json.load(f)
    out = []
    for vid in ("1", "2"):
        v = val[f"video_{vid}"]
        out.append(dict(
            id=int(vid),
            motion=MOTIONS[vid],
            t=v["t"],
            n_corners=v["n_corners"],
            lk_mean_err=v["lk_mean_err"],
            lk_median_err=v["lk_median_err"],
            dense_grid_mee=v["dense_grid_mee"],
            pair_image=f"/static/flow_pair_{vid}.png",
            flow_gif=f"/static/flow_{vid}.gif",
            flow_video=f"/static/flow_viz_{vid}.mp4",
        ))
    return out


def sfm_summary() -> dict:
    with open(DATA / "sfm_data.json") as f:
        data = json.load(f)
    K = np.array(data["K"])
    pts_true = np.array(data["object_points"])          # N x 3, Z = 0
    boundary_idx = data["boundary_indices"]
    n_views = len(data["cameras"])

    cameras = []
    Hs_est = []
    for cam in data["cameras"]:
        R = np.array(cam["R"]); t = np.array(cam["t"]); C = np.array(cam["C"])
        H_true = sfm.true_homography(K, R, t)
        H_est = sfm.homography_dlt(pts_true[:, :2], np.array(cam["projections"]))
        Hs_est.append(H_est)
        cameras.append(dict(view=len(cameras) + 1,
                            C=[round(float(x), 1) for x in C],
                            homography_rel_err=sfm.rel_error(H_est, H_true)))

    uvs = [np.array(c["projections"]) for c in data["cameras"]]
    rec_planar = sfm.planar_reconstruct(Hs_est, uvs)
    Ps = [np.array(c["P"]) for c in data["cameras"]]
    rec_tri = np.array([sfm.triangulate(Ps, [uvs[v][j] for v in range(n_views)])
                        for j in range(len(pts_true))])

    err_planar = np.linalg.norm(rec_planar - pts_true[:, :2], axis=1)
    err_tri = np.linalg.norm(rec_tri - pts_true, axis=1)

    b = boundary_idx

    def area2d(p):
        x, y = p[:, 0], p[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    return dict(
        K=K.tolist(),
        focal=float(K[0, 0]),
        principal_point=[float(K[0, 2]), float(K[1, 2])],
        n_points=len(pts_true),
        n_boundary=len(b),
        n_interior=len(pts_true) - len(b),
        cameras=cameras,
        views=[f"/static/sfm_view_{v + 1}.png" for v in range(n_views)],
        reconstruction="/static/sfm_reconstruction.png",
        planar_mean_err=float(err_planar.mean()),
        planar_max_err=float(err_planar.max()),
        tri_mean_err=float(err_tri.mean()),
        tri_max_err=float(err_tri.max()),
        tri_z_min=float(rec_tri[:, 2].min()),
        tri_z_max=float(rec_tri[:, 2].max()),
        boundary_area_true=float(area2d(pts_true[b, :2])),
        boundary_area_rec=float(area2d(rec_planar[b])),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regenerate", action="store_true",
                    help="re-run the full pipeline before assembling")
    args = ap.parse_args()

    if args.regenerate:
        run_pipeline()

    copy_assets()
    make_gif(1, "flow_1.gif")
    make_gif(2, "flow_2.gif")

    results = dict(optical_flow=optical_flow_summary(), sfm=sfm_summary())
    with open(STATIC / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {STATIC / 'results.json'}")


if __name__ == "__main__":
    main()
