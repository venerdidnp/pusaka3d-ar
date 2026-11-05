"""Train the from-scratch VoxelGrid to reconstruct the toy Pusaka3D temple
from the 8-view capture.json dataset, then render a genuinely held-out
novel view and compare it to the analytic ground truth.

This is the from-scratch stand-in for `ns-train nerfacto` (see
Module5_EN.tex's Software section for the real nerfstudio command) --
matched against nerfstudio's renderer is not possible on this authoring
host (no GPU/COLMAP pipeline run here); matched against the *analytic*
ground truth (which generated the training images in the first place) is
the reconciliation this script performs instead. See README.md.
"""
from __future__ import annotations

import json
import os

import numpy as np

from capture import DATA_DIR, save_ppm
from nerf_toy import VoxelGrid, render_analytic, render_grid, train, ring_camera, psnr

RESOLUTION = 12
LR = 100.0
N_ITERS = 800
NOVEL_ANGLE_DEG = 22.5  # exactly between two training views (0 and 45) -- a fair held-out test


def load_capture(data_dir=DATA_DIR):
    with open(os.path.join(data_dir, "poses.json")) as f:
        manifest = json.load(f)
    views, targets = [], []
    for frame in manifest["frames"]:
        pos = np.array(frame["position"])
        c2w = np.array(frame["camera_to_world_rotation"])
        views.append((pos, c2w))
        target = render_analytic(manifest["width"], manifest["height"], manifest["fov_deg"],
                                  pos, c2w, manifest["near"], manifest["far"], manifest["n_samples"],
                                  extended=manifest.get("extended", False))
        targets.append(target)
    return manifest, views, targets


def main():
    manifest, views, targets = load_capture()
    W, H, FOV = manifest["width"], manifest["height"], manifest["fov_deg"]
    near, far, n_samples = manifest["near"], manifest["far"], manifest["n_samples"]

    grid = VoxelGrid(resolution=RESOLUTION, seed=0)
    print(f"Training a {RESOLUTION}^3 voxel grid on {len(views)} views "
          f"({W}x{H} px, {n_samples} samples/ray), lr={LR}, {N_ITERS} iters ...")
    history = train(grid, views, W, H, FOV, targets, near, far, n_samples, lr=LR, n_iters=N_ITERS, log_every=200)
    for it, mse in history:
        print(f"  iter {it:4d}   train MSE = {mse:.5f}")

    train_psnrs = []
    for (cam_pos, c2w), target in zip(views, targets):
        pred, _ = render_grid(grid, W, H, FOV, cam_pos, c2w, near, far, n_samples)
        train_psnrs.append(psnr(pred, target))
    print(f"mean training-view PSNR = {np.mean(train_psnrs):.2f} dB")

    novel_cam, novel_c2w = ring_camera(NOVEL_ANGLE_DEG, radius=manifest["radius"], height=manifest["camera_height"])
    novel_target = render_analytic(W, H, FOV, novel_cam, novel_c2w, near, far, n_samples)
    novel_pred, _ = render_grid(grid, W, H, FOV, novel_cam, novel_c2w, near, far, n_samples)
    novel_psnr = psnr(novel_pred, novel_target)
    print(f"held-out novel view ({NOVEL_ANGLE_DEG} deg, between two training views): PSNR = {novel_psnr:.2f} dB")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    os.makedirs(out_dir, exist_ok=True)
    save_ppm(os.path.join(out_dir, "novel_view_target.ppm"), novel_target)
    save_ppm(os.path.join(out_dir, "novel_view_predicted.ppm"), novel_pred)
    diff = np.clip(np.abs(novel_pred - novel_target) * 4.0, 0, 1)
    save_ppm(os.path.join(out_dir, "novel_view_diff_x4.ppm"), diff)
    print(f"wrote out/novel_view_target.ppm, out/novel_view_predicted.ppm, out/novel_view_diff_x4.ppm")

    return grid, train_psnrs, novel_psnr


if __name__ == "__main__":
    main()
