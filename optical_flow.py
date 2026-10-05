"""
optical_flow.py — compute, visualize and validate optical flow
===============================================================

For each of the two synthetic videos produced by ``synthetic_data.py`` this
script:

  1. computes **dense** optical flow between every consecutive frame pair with
     Gunnar Farneback's algorithm and renders the standard Middlebury
     colour-wheel visualization to a video
     (hue = motion direction, saturation/value = magnitude);
  2. computes **sparse** optical flow with Lucas-Kanade pyramidal tracking of
     Shi-Tomasi corners;
  3. **validates** the tracking: for two consecutive frames it compares the
     LK-tracked pixel location against the *analytic* ground-truth displacement
     from the known motion model (imported from ``synthetic_data``) and reports
     the endpoint error;
  4. quantifies dense-flow accuracy against ground truth (mean endpoint error
     over a pixel grid) as further evidence.

Outputs (written into ``data/``)
--------------------------------
  flow_viz_1.mp4 / flow_viz_2.mp4     colour-wheel flow videos
  flow_pair_1.png / flow_pair_2.png   montage of the two frames + flow + arrows
  validation.json                     numeric validation results

Run:  python optical_flow.py
"""

import json
import os
import numpy as np
import cv2

import matplotlib
matplotlib.use("Agg")                 # headless backend (no display needed)
import matplotlib.pyplot as plt

import synthetic_data as sd

OUT_DIR = "data"

FARNEBACK_PARAMS = dict(pyr_scale=0.5, levels=3, winsize=15,
                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

# frames used for the "two consecutive frames" validation (5 s into the clip)
VALIDATION_T = 150


# ----------------------------------------------------------------------------
# Colour-wheel visualization (Middlebury convention)
# ----------------------------------------------------------------------------
def flow_to_color(flow):
    """flow: (H, W, 2) float array of (dx, dy) -> BGR colour image."""
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = (ang / 2.0).astype(np.uint8)          # hue = direction
    hsv[..., 1] = 255                                   # full saturation
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _write_color_video(path, frames, fps=sd.FPS, size=(sd.W, sd.H)):
    os.makedirs(OUT_DIR, exist_ok=True)
    vw = None
    for fourcc_code, ext in (("mp4v", ".mp4"), ("MJPG", ".avi")):
        out_path = os.path.splitext(path)[0] + ext
        fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
        vw = cv2.VideoWriter(out_path, fourcc, fps, size, isColor=True)
        if vw.isOpened():
            break
    n = 0
    for f in frames:
        vw.write(f)
        n += 1
    vw.release()
    return out_path, n


# ----------------------------------------------------------------------------
# Dense flow over the whole clip -> colour-wheel video
# ----------------------------------------------------------------------------
def render_flow_video(video_id):
    cap = cv2.VideoCapture(sd.video_path(video_id))
    ok, prev = cap.read()
    if not ok:
        raise RuntimeError(f"could not read video_{video_id}")
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    def frame_iter():
        nonlocal prev
        while True:
            ok, nxt = cap.read()
            if not ok:
                break
            nxt = cv2.cvtColor(nxt, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev, nxt, None,
                                                **FARNEBACK_PARAMS)
            yield flow_to_color(flow)
            prev = nxt

    out, n = _write_color_video(f"{OUT_DIR}/flow_viz_{video_id}.mp4",
                                frame_iter())
    cap.release()
    return out, n


# ----------------------------------------------------------------------------
# Validation on a single pair of consecutive frames
# ----------------------------------------------------------------------------
def analyze_pair(video_id, t=VALIDATION_T):
    cap = cv2.VideoCapture(sd.video_path(video_id))
    cap.set(cv2.CAP_PROP_POS_FRAMES, t)
    ok, f0 = cap.read()
    ok2, f1 = cap.read()
    cap.release()
    if not (ok and ok2):
        raise RuntimeError(f"could not read frames {t}, {t+1} of video_{video_id}")
    g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)

    # --- dense flow (Farneback) --------------------------------------------
    flow = cv2.calcOpticalFlowFarneback(g0, g1, None, **FARNEBACK_PARAMS)
    color = flow_to_color(flow)

    # --- sparse flow (Shi-Tomasi corners -> Lucas-Kanade pyramid) ----------
    p0 = cv2.goodFeaturesToTrack(g0, maxCorners=400, qualityLevel=0.01,
                                 minDistance=10)
    p1, status, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None,
                                             winSize=(21, 21), maxLevel=3,
                                             criteria=(cv2.TERM_CRITERIA_EPS
                                                       | cv2.TERM_CRITERIA_COUNT,
                                                       30, 0.01))
    ok_idx = status.reshape(-1).astype(bool)
    p0 = p0[ok_idx].reshape(-1, 2)
    p1 = p1[ok_idx].reshape(-1, 2)

    # --- validate: compare LK result to analytic ground truth ----------------
    errs = []
    rows = []
    for (x0, y0), (x1, y1) in zip(p0, p1):
        gtx, gty = sd.gt_flow_at(video_id, t, x0, y0)
        lk_dx, lk_dy = x1 - x0, y1 - y0
        e = float(np.hypot(lk_dx - gtx, lk_dy - gty))
        errs.append(e)
        rows.append(dict(x=float(x0), y=float(y0),
                         gt=(gtx, gty), lk=(lk_dx, lk_dy), err=e))
    errs = np.array(errs)
    n_pts = len(errs)

    # --- dense MEE against ground truth on a grid ----------------------------
    step = 8
    ys, xs = np.mgrid[step // 2:sd.H:step, step // 2:sd.W:step]
    mee = 0.0
    cnt = 0
    for i in range(ys.shape[0]):
        for j in range(xs.shape[1]):
            x, y = xs[i, j], ys[i, j]
            gtx, gty = sd.gt_flow_at(video_id, t, x, y)
            fx, fy = flow[y, x]
            mee += np.hypot(fx - gtx, fy - gty)
            cnt += 1
    mee /= cnt

    result = dict(video=video_id, t=t, n_corners=int(n_pts),
                  lk_mean_err=float(errs.mean()),
                  lk_median_err=float(np.median(errs)),
                  dense_grid_mee=float(mee))

    # --- build a 2x2 montage for the report ----------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    ax[0, 0].imshow(g0, cmap="gray"); ax[0, 0].set_title(f"frame t = {t}")
    ax[0, 1].imshow(g1, cmap="gray"); ax[0, 1].set_title(f"frame t = {t+1}")
    ax[1, 0].imshow(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
    ax[1, 0].set_title("dense flow (Farneback) — hue=direction, val=magnitude")
    ax[1, 1].imshow(g0, cmap="gray")
    # draw LK measured (green) vs ground-truth (red) displacement arrows
    for (x0, y0), (x1, y1) in zip(p0, p1):
        gtx, gty = sd.gt_flow_at(video_id, t, x0, y0)
        ax[1, 1].arrow(x0, y0, x1 - x0, y1 - y0,
                       color="lime", head_width=3, length_includes_head=True,
                       alpha=0.7)
        ax[1, 1].arrow(x0, y0, gtx, gty, color="red",
                       head_width=3, length_includes_head=True, alpha=0.6)
    ax[1, 1].set_title("LK tracked (green) vs ground truth (red)")
    for a in ax.ravel():
        a.axis("off")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/flow_pair_{video_id}.png", dpi=120)
    plt.close(fig)

    return result, rows


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_results = {}
    for vid in (1, 2):
        print(f"--- video {vid} ---")
        out, n = render_flow_video(vid)
        print(f"  flow visualization video: {out} ({n} frames)")
        res, _ = analyze_pair(vid)
        all_results[f"video_{vid}"] = res
        print(f"  corners tracked: {res['n_corners']}")
        print(f"  LK mean endpoint error vs ground truth: "
              f"{res['lk_mean_err']:.3f} px "
              f"(median {res['lk_median_err']:.3f} px)")
        print(f"  dense (Farneback) grid MEE vs ground truth: "
              f"{res['dense_grid_mee']:.3f} px")

    with open(f"{OUT_DIR}/validation.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("wrote data/validation.json")


if __name__ == "__main__":
    main()
