"""
synthetic_data.py — generate synthetic input data for Assignment 6
===================================================================

Purpose
-------
Assignment 6 needs (a) two >= 30 s videos with motion for the optical-flow /
tracking part, and (b) four viewpoints of a planar object for the
structure-from-motion part.  We generate everything *synthetically* so that
every "measured" quantity has an exact, known ground truth to validate against
(no hand-labelling, no camera-calibration error).

Outputs (written into ``data/``)
--------------------------------
  video_1.mp4       30 s @ 30 fps, 640x480 -- horizontal camera pan over a
                    textured background + a textured disk orbiting the centre.
  video_2.mp4       30 s @ 30 fps, 640x480 -- dolly zoom in/out (radial flow,
                    focus of expansion/contraction at the centre) + a textured
                    square oscillating horizontally.
  sfm_view_<1..4>.png   the planar object rendered from four calibrated cameras.
  sfm_data.json     camera intrinsics K, extrinsics (R, t, centre C), the
                    ground-truth 3-D object points, the ordered boundary
                    indices, and every camera's 2-D projections.

The motion models are exposed as functions (``PAN_SPEED``, ``zoom_scale``,
``ground_truth_flow``, ``disk_flow``, ``square_flow``, ``video_path``) so that
``optical_flow.py`` can recompute the *analytic* ground truth for the tracking
validation -- one single source of truth.

Usage
-----
    python synthetic_data.py
"""

import json
import os
import numpy as np
import cv2

# ----------------------------------------------------------------------------
# Global constants
# ----------------------------------------------------------------------------
W, H = 640, 480                     # image size (px)
FPS = 30
DURATION_S = 30
N_FRAMES = FPS * DURATION_S         # 900 frames

OUT_DIR = "data"

# ------------------------- Video 1: pan + orbiting disk ----------------------
PAN_SPEED = 3.0                     # background translation, px/frame (leftward)
DISK_RADIUS = 45                    # px
DISK_ORBIT_A = 90.0                 # horizontal orbit amplitude, px
DISK_ORBIT_B = 70.0                 # vertical orbit amplitude, px
DISK_OMEGA = 2.0 * np.pi * 2.0 / N_FRAMES   # ~2 full orbits over the clip

# ------------------------- Video 2: dolly zoom + translating square -----------
ZOOM_AMPLITUDE = 0.5                # zoom oscillates 1 +/- 0.5 (bounded dolly)
ZOOM_PERIOD = N_FRAMES / 3.0        # frames per in/out cycle (~10 s -> 3 cycles)
SQUARE_SIZE = 60                    # px
SQUARE_AMP = 220.0                  # horizontal oscillation amplitude, px
SQUARE_OMEGA = 2.0 * np.pi * 2.0 / N_FRAMES   # 2 full oscillations over the clip
SQUARE_Y0 = 0.6 * H                 # vertical position of the square centre

# ------------------------- Structure-from-motion scene ------------------------
FOCAL = 600.0                       # focal length, px
CX, CY = W / 2.0, H / 2.0           # principal point
K = np.array([[FOCAL, 0, CX],
              [0, FOCAL, CY],
              [0, 0, 1]], dtype=np.float64)


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def _write_video(path, frame_iter, fps=FPS, size=(W, H)):
    """Write a grayscale frame generator to disk, preferring mp4v then MJPG."""
    os.makedirs(OUT_DIR, exist_ok=True)
    vw = None
    for fourcc_code, ext in (("mp4v", ".mp4"), ("MJPG", ".avi")):
        out_path = os.path.splitext(path)[0] + ext
        fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
        vw = cv2.VideoWriter(out_path, fourcc, fps, size, isColor=False)
        if vw.isOpened():
            break
    if vw is None or not vw.isOpened():
        raise RuntimeError("no usable VideoWriter codec found")
    n = 0
    for frame in frame_iter:
        vw.write(frame)
        n += 1
    vw.release()
    return out_path, n


def video_path(video_id):
    """Return the path of the generated video (handles .mp4 / .avi fallback)."""
    for ext in (".mp4", ".avi"):
        p = f"{OUT_DIR}/video_{video_id}{ext}"
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"video_{video_id} not found; run `python synthetic_data.py` first")


# ----------------------------------------------------------------------------
# Video 1 — horizontal pan + orbiting textured disk
# ----------------------------------------------------------------------------
def _video1_frames():
    rng = np.random.default_rng(1)
    # a wide random texture; we slide a W-px window across it to make the pan.
    pan_px = int(PAN_SPEED * N_FRAMES) + W + 16
    tex = rng.integers(0, 256, size=(H, pan_px)).astype(np.float32)
    tex = cv2.GaussianBlur(tex, (7, 7), 0)      # soften so flow is well defined
    Y, X = np.mgrid[0:H, 0:W].astype(np.float32)

    for t in range(N_FRAMES):
        # --- background: crop a sliding window -------------------------------
        x0 = int(round(PAN_SPEED * t))
        bg = tex[:, x0:x0 + W].astype(np.uint8).copy()

        # --- disk: centre follows an ellipse about the image centre ----------
        cx = W / 2.0 + DISK_ORBIT_A * np.cos(DISK_OMEGA * t)
        cy = H / 2.0 + DISK_ORBIT_B * np.sin(DISK_OMEGA * t)
        mask = (X - cx) ** 2 + (Y - cy) ** 2 <= DISK_RADIUS ** 2
        # checker texture anchored to the disk centre -> rigid body motion
        parity = (np.floor((X - cx) / 12.0).astype(np.int64)
                  + np.floor((Y - cy) / 12.0).astype(np.int64)) & 1
        check = np.where(parity == 1, 190, 70).astype(np.uint8)
        bg[mask] = check[mask]
        yield bg


# ----------------------------------------------------------------------------
# Video 2 — dolly zoom (in/out) + horizontally oscillating textured square
# ----------------------------------------------------------------------------
def _video2_frames():
    rng = np.random.default_rng(2)
    tex = rng.integers(0, 256, size=(2 * H, 2 * W)).astype(np.float32)
    tex = cv2.GaussianBlur(tex, (7, 7), 0)
    Y, X = np.mgrid[0:H, 0:W].astype(np.float32)

    for t in range(N_FRAMES):
        # --- background: sub-pixel-exact centred zoom ------------------------
        # Sample the texture at scale 1/s about its centre.  warpAffine treats
        # M as the FORWARD (src -> dst) map and inverts it internally, so
        # M = [[s,0,W(0.5-s)],[0,s,H(0.5-s)]] makes dst (x,y) read the texture
        # at (W + (x-W/2)/s, H + (y-H/2)/s) -- a pure scale about the display
        # centre (W/2,H/2).  The analytic flow (x-CX, y-CY)*(s'/s - 1) is exact.
        s = zoom_scale(t)
        M = np.float32([[s, 0, W * (0.5 - s)],
                        [0, s, H * (0.5 - s)]])
        bg = cv2.warpAffine(tex, M, (W, H), flags=cv2.INTER_LINEAR).astype(np.uint8).copy()

        # --- square: horizontal oscillation (bounded, faster motion) --------
        sqx, sqy = square_center(t)
        hs = SQUARE_SIZE // 2
        x0, x1 = int(round(sqx - hs)), int(round(sqx + hs))
        y0, y1 = int(round(sqy - hs)), int(round(sqy + hs))
        if 0 <= x0 and x1 <= W and 0 <= y0 and y1 <= H:
            xx = X[y0:y1, x0:x1] - sqx
            yy = Y[y0:y1, x0:x1] - sqy
            par = (np.floor(xx / 10.0).astype(np.int64)
                   + np.floor(yy / 10.0).astype(np.int64)) & 1
            bg[y0:y1, x0:x1] = np.where(par == 1, 200, 60).astype(np.uint8)
        yield bg


# ----------------------------------------------------------------------------
# Analytic ground-truth motion (used by optical_flow.py to validate tracking)
# ----------------------------------------------------------------------------
def zoom_scale(t):
    """Bounded dolly: the zoom factor oscillates sinusoidally (zoom in, then
    out) so that (a) per-frame flow stays measurable and (b) the texture is
    never magnified beyond its own resolution.  Zooming in gives a diverging
    (focus of expansion) field, zooming out a converging (focus of contraction)."""
    return 1.0 + ZOOM_AMPLITUDE * np.sin(2.0 * np.pi * t / ZOOM_PERIOD)


def ground_truth_flow(video, t, x, y):
    """Analytic (dx, dy) of a *background* scene point between frame t and t+1.

    video: 1 or 2.  (x, y): image location in frame t.  Returns (dx, dy).
    """
    if video == 1:
        return (-PAN_SPEED, 0.0)                 # background pan (leftward)
    elif video == 2:
        s0, s1 = zoom_scale(t), zoom_scale(t + 1)
        f = s1 / s0 - 1.0                        # outward radial flow
        return (f * (x - CX), f * (y - CY))
    raise ValueError("video must be 1 or 2")


def disk_center(t):
    return (W / 2.0 + DISK_ORBIT_A * np.cos(DISK_OMEGA * t),
            H / 2.0 + DISK_ORBIT_B * np.sin(DISK_OMEGA * t))


def disk_flow(t):
    c0, c1 = disk_center(t), disk_center(t + 1)
    return (c1[0] - c0[0], c1[1] - c0[1])


def square_center(t):
    return (W / 2.0 + SQUARE_AMP * np.sin(SQUARE_OMEGA * t), SQUARE_Y0)


def square_flow(t):
    c0, c1 = square_center(t), square_center(t + 1)
    return (c1[0] - c0[0], 0.0)


def gt_flow_at(video, t, x, y):
    """Ground-truth displacement for an arbitrary pixel (background or object)."""
    x, y = float(x), float(y)
    if video == 1:
        cx, cy = disk_center(t)
        if (x - cx) ** 2 + (y - cy) ** 2 <= DISK_RADIUS ** 2:
            return disk_flow(t)
        return ground_truth_flow(1, t, x, y)
    elif video == 2:
        sx, sy = square_center(t)
        if abs(x - sx) <= SQUARE_SIZE // 2 and abs(y - sy) <= SQUARE_SIZE // 2:
            return square_flow(t)
        return ground_truth_flow(2, t, x, y)
    raise ValueError("video must be 1 or 2")


# ----------------------------------------------------------------------------
# Structure-from-motion scene: planar object + four calibrated cameras
# ----------------------------------------------------------------------------
BOUNDARY_XY = np.array([
    [-45.0, -25.0], [-15.0, -25.0], [-15.0, -5.0], [5.0, -5.0],
    [5.0, -25.0], [45.0, -25.0], [45.0, 15.0], [15.0, 15.0],
    [15.0, 35.0], [-45.0, 35.0],
], dtype=np.float64)                     # ordered boundary, world mm, Z = 0

INTERIOR_XY = np.array([
    [-30.0, 0.0], [-30.0, 15.0], [-10.0, 15.0], [-10.0, 0.0],
    [0.0, 15.0], [25.0, -10.0], [25.0, 0.0], [-30.0, -15.0],
    [0.0, -15.0], [25.0, 10.0],
], dtype=np.float64)                      # extra interior samples


def make_object_points():
    pts2 = np.vstack([BOUNDARY_XY, INTERIOR_XY])
    n = len(pts2)
    pts3 = np.hstack([pts2, np.zeros((n, 1))])      # lift to Z = 0 plane
    return pts3, list(range(len(BOUNDARY_XY)))      # (N x 3), boundary indices


def look_at(C, target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, -1.0, 0.0])):
    """Rotation R and translation t of a camera at C looking at `target`."""
    C = np.asarray(C, dtype=np.float64)
    f = target - C
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)                    # unit because s, f are orthonormal
    R = np.stack([s, u, f], axis=0)       # rows = camera axes in world coords
    t = -R @ C
    return R, t


# Four distinct camera centres (world mm); each looks at the origin.
CAMERAS = [
    dict(C=np.array([0.0, 0.0, 500.0])),          # frontal
    dict(C=np.array([-300.0, 40.0, 480.0])),      # left, slightly high
    dict(C=np.array([300.0, -40.0, 520.0])),      # right, slightly low
    dict(C=np.array([0.0, 200.0, 450.0])),        # high, closer
]


def project(P, X_h):
    """Project a homogeneous 3-D point X_h (4,) with camera P (3x4)."""
    x = P @ X_h
    return x[0] / x[2], x[1] / x[2]


def draw_view(P, pts3, boundary_idx):
    """Render the object points + boundary outline into a 640x480 image."""
    img = np.zeros((H, W), np.uint8)
    uvs = np.array([project(P, np.append(p, 1.0)) for p in pts3])
    b = boundary_idx
    poly = np.vstack([uvs[b], uvs[b[0]]]).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [poly], True, 255, 2)
    for (u, v) in uvs:
        cv2.circle(img, (int(round(u)), int(round(v))), 3, 255, -1)
    return img, uvs


def generate_sfm():
    pts3, boundary_idx = make_object_points()
    data = dict(K=K.tolist(),
                object_points=pts3.tolist(),
                boundary_indices=boundary_idx,
                cameras=[])
    for i, cam in enumerate(CAMERAS):
        R, t = look_at(cam["C"])
        C_cam = -R.T @ t                                    # camera centre
        P = K @ np.hstack([R, t.reshape(-1, 1)])
        img, uvs = draw_view(P, pts3, boundary_idx)
        cv2.imwrite(f"{OUT_DIR}/sfm_view_{i + 1}.png", img)
        data["cameras"].append(dict(R=R.tolist(), t=t.tolist(),
                                    C=C_cam.tolist(), P=P.tolist(),
                                    projections=uvs.tolist()))
    with open(f"{OUT_DIR}/sfm_data.json", "w") as f:
        json.dump(data, f, indent=2)
    return data


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("generating video 1 (pan + orbiting disk) ...")
    p1, n1 = _write_video(f"{OUT_DIR}/video_1.mp4", _video1_frames())
    print(f"  wrote {p1} ({n1} frames)")

    print("generating video 2 (zoom + translating square) ...")
    p2, n2 = _write_video(f"{OUT_DIR}/video_2.mp4", _video2_frames())
    print(f"  wrote {p2} ({n2} frames)")

    print("generating structure-from-motion viewpoints ...")
    generate_sfm()
    print("  wrote data/sfm_view_1..4.png and data/sfm_data.json")

    print("done.")


if __name__ == "__main__":
    main()
