"""
structure_from_motion.py — reconstruct a planar object from 4 viewpoints
=========================================================================

Recovers the 3-D (planar) structure of the object rendered by
``synthetic_data.py`` and validates it against known ground truth.

Two complementary reconstructions are demonstrated:

  1. PLANAR HOMOGRAPHY (the natural method for a Z = 0 object):
       - estimate the world-plane -> image homography H for each view with the
         Direct Linear Transform (DLT);
       - verify the estimated H equals K [r1 r2 t] from the camera model;
       - invert H to map each image point back to world-plane coordinates and
         average the four views.

  2. MULTI-VIEW TRIANGULATION (the general SfM pipeline):
       - build each camera matrix P = K [R | t];
       - linearly triangulate each point from all four views (DLT null-space).

Both results are compared against ground truth (mean world-space error), and the
reconstructed boundary is drawn next to the true boundary.

Outputs (written into ``data/``)
--------------------------------
  sfm_reconstruction.png   figure: 4 viewpoints + true vs reconstructed boundary

Run:  python structure_from_motion.py
"""

import json
import os
import numpy as np
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import synthetic_data as sd

OUT_DIR = "data"


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------
def homography_dlt(world_xy, image_uv):
    """Estimate the homography mapping world-plane (X,Y) -> image (u,v) by DLT.

    Solves  A h = 0  with A built from the cross product  [u,v,1] x H [X,Y,1] = 0.
    """
    n = len(world_xy)
    A = np.zeros((2 * n, 9))
    for i in range(n):
        X, Y = world_xy[i]
        u, v = image_uv[i]
        A[2 * i]     = [0, 0, 0, -X, -Y, -1, v * X, v * Y, v]
        A[2 * i + 1] = [X, Y, 1, 0, 0, 0, -u * X, -u * Y, -u]
    _, _, Vt = np.linalg.svd(A)
    return Vt[-1].reshape(3, 3)


def true_homography(K, R, t):
    """Ground-truth plane-to-image homography  K [r1 r2 t]."""
    return K @ np.column_stack([R[:, 0], R[:, 1], t])


def rel_error(A, B):
    """Scale- and sign-invariant relative Frobenius error between two 3x3 matrices.

    Homographies are only determined up to a (possibly negative) scale factor,
    so we compare against both signs and keep the smaller.
    """
    A = A / np.linalg.norm(A)
    B = B / np.linalg.norm(B)
    return float(min(np.linalg.norm(A - B), np.linalg.norm(A + B)))


def triangulate(Ps, xs):
    """Linear (DLT) triangulation of a point from several views.

    Ps: list of 3x4 camera matrices; xs: list of (u, v) image coordinates.
    """
    n = len(Ps)
    A = np.zeros((2 * n, 4))
    for i in range(n):
        P, (u, v) = Ps[i], xs[i]
        A[2 * i]     = u * P[2] - P[0]
        A[2 * i + 1] = v * P[2] - P[1]
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]


def planar_reconstruct(Hs, uvs_per_view):
    """Recover world-plane (X, Y) by inverting each view's homography, averaged."""
    n_views = len(Hs)
    n_pts = len(uvs_per_view[0])
    rec = np.zeros((n_pts, 2))
    for j in range(n_pts):
        acc = np.zeros(2)
        for v in range(n_views):
            wh = np.linalg.inv(Hs[v]) @ np.array([*uvs_per_view[v][j], 1.0])
            acc += wh[:2] / wh[2]
        rec[j] = acc / n_views
    return rec


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/sfm_data.json") as f:
        data = json.load(f)

    K = np.array(data["K"])
    pts_true = np.array(data["object_points"])         # N x 3, Z = 0
    boundary_idx = data["boundary_indices"]
    n_views = len(data["cameras"])

    # --- print camera parameters / positions (required by the assignment) -----
    print("=" * 78)
    print("Camera parameters (intrinsics K) and positions (extrinsics R, t, C)")
    print("=" * 78)
    print(f"K =\n{K}")
    print(f"object points (world mm, Z = 0): {len(pts_true)} "
          f"({len(boundary_idx)} boundary + {len(pts_true) - len(boundary_idx)} interior)")
    for i, cam in enumerate(data["cameras"]):
        R = np.array(cam["R"]); t = np.array(cam["t"]); C = np.array(cam["C"])
        print(f"\nView {i+1}: centre C = {np.round(C, 1)} mm")
        print(f"  R =\n{np.round(R, 4)}")
        print(f"  t = {np.round(t, 1)} mm")

    # ======================= 1. planar homography ============================
    Hs_est = []
    for i, cam in enumerate(data["cameras"]):
        R = np.array(cam["R"]); t = np.array(cam["t"])
        H_true = true_homography(K, R, t)
        H_est = homography_dlt(pts_true[:, :2], np.array(cam["projections"]))
        Hs_est.append(H_est)
        print(f"\nView {i+1} homography: ||H_est - H_true||_F / ||H_true||_F "
              f"(normalized) = {rel_error(H_est, H_true):.6f}")

    uvs_per_view = [np.array(c["projections"]) for c in data["cameras"]]
    rec_planar = planar_reconstruct(Hs_est, uvs_per_view)     # N x 2

    # ======================= 2. triangulation ================================
    Ps = [np.array(c["P"]) for c in data["cameras"]]
    rec_tri = np.array([triangulate(Ps, [uvs_per_view[v][j] for v in range(n_views)])
                        for j in range(len(pts_true))])          # N x 3

    # --- errors ---------------------------------------------------------------
    err_planar = np.linalg.norm(rec_planar - pts_true[:, :2], axis=1)
    err_tri = np.linalg.norm(rec_tri - pts_true, axis=1)
    print("\n" + "=" * 78)
    print("Reconstruction error vs ground truth")
    print("=" * 78)
    print(f"planar homography : mean {err_planar.mean():.4f} mm, "
          f"max {err_planar.max():.4f} mm")
    print(f"4-view triangulation: mean {err_tri.mean():.4f} mm, "
          f"max {err_tri.max():.4f} mm")
    print(f"triangulated Z range: [{rec_tri[:, 2].min():.4f}, "
          f"{rec_tri[:, 2].max():.4f}] mm (true Z = 0)")

    # --- boundary reconstruction ----------------------------------------------
    b = boundary_idx
    # polygon area (shoelace) for true vs reconstructed boundaries
    def area2d(pts):
        x, y = pts[:, 0], pts[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    a_true = area2d(pts_true[b, :2])
    a_rec = area2d(rec_planar[b])
    print(f"boundary area: true {a_true:.1f} mm^2 vs reconstructed "
          f"{a_rec:.1f} mm^2 (rel. err {(a_rec - a_true) / a_true * 100:.3f}%)")

    # --- figure ---------------------------------------------------------------
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    for v in range(n_views):
        r, c = divmod(v, 3)
        img = cv2.imread(f"{OUT_DIR}/sfm_view_{v+1}.png", cv2.IMREAD_GRAYSCALE)
        ax[r, c].imshow(img, cmap="gray")
        ax[r, c].set_title(f"view {v+1}  C={np.round(np.array(data['cameras'][v]['C']), 0)}")
        ax[r, c].axis("off")
    # true vs reconstructed boundary
    axb = ax[1, 1]
    axb.plot(*pts_true[b, :2].T, "o-", color="black", label="true")
    axb.plot(*rec_planar[b].T, "x--", color="crimson", label="planar homography")
    axb.plot(*rec_tri[b, :2].T, "+:", color="royalblue", label="triangulation")
    axb.set_aspect("equal"); axb.legend(); axb.set_title("boundary (world mm)")
    # reconstructed point cloud (triangulation), coloured by Z (should be ~0)
    axc = ax[1, 2]
    sc = axc.scatter(rec_tri[:, 0], rec_tri[:, 1], c=rec_tri[:, 2], cmap="viridis")
    axc.set_aspect("equal"); fig.colorbar(sc, ax=axc, label="Z (mm)")
    axc.set_title("triangulated point cloud")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/sfm_reconstruction.png", dpi=130)
    plt.close(fig)
    print("wrote data/sfm_reconstruction.png")


if __name__ == "__main__":
    main()
