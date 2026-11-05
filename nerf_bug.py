"""Diagnose-and-fix: too few / too-clustered training views cause a NeRF to
memorise the views it saw and fail on angles it did not -- the toy-scale,
from-scratch analogue of real "floaters" / background-collapse when a
capture has too few poses or too narrow a baseline.

BUGGY:  2 views, 20 deg apart (0 deg, 20 deg) -- almost the same viewpoint.
FIXED:  8 views spread evenly around the full ring (0..315 deg, step 45).

Both are trained identically otherwise and evaluated on the *same* held-out
novel view, chosen to be far from the buggy pair's narrow baseline. The gap
in novel-view PSNR is the diagnostic evidence: it is not that the buggy grid
"looks wrong" in general -- it fits its own two training views about as well
as the fixed grid fits its eight -- it is that it has never seen, and cannot
predict, the rest of the object.
"""
from __future__ import annotations

import os

import numpy as np

from capture import save_ppm
from nerf_toy import VoxelGrid, render_analytic, render_grid, train, ring_camera, psnr

W, H, FOV = 20, 16, 50.0
NEAR, FAR, N_SAMPLES = 1.0, 5.0, 32
RADIUS, CAM_HEIGHT = 3.0, 1.0
LR, N_ITERS = 100.0, 500
NOVEL_ANGLE_DEG = 190.0  # far from the buggy cluster (0,20); a fair stress test either way

BUGGY_ANGLES = [0, 20]
FIXED_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]


def _train_on(angles, seed=0):
    views = [ring_camera(a, radius=RADIUS, height=CAM_HEIGHT) for a in angles]
    targets = [render_analytic(W, H, FOV, cam, c2w, NEAR, FAR, N_SAMPLES) for cam, c2w in views]
    grid = VoxelGrid(resolution=12, seed=seed)
    history = train(grid, views, W, H, FOV, targets, NEAR, FAR, N_SAMPLES, lr=LR, n_iters=N_ITERS, log_every=N_ITERS - 1)
    train_psnrs = [psnr(render_grid(grid, W, H, FOV, cam, c2w, NEAR, FAR, N_SAMPLES)[0], t)
                   for (cam, c2w), t in zip(views, targets)]
    return grid, history, float(np.mean(train_psnrs))


def diagnose_and_fix():
    novel_cam, novel_c2w = ring_camera(NOVEL_ANGLE_DEG, radius=RADIUS, height=CAM_HEIGHT)
    novel_target = render_analytic(W, H, FOV, novel_cam, novel_c2w, NEAR, FAR, N_SAMPLES)

    buggy_grid, buggy_hist, buggy_train_psnr = _train_on(BUGGY_ANGLES)
    fixed_grid, fixed_hist, fixed_train_psnr = _train_on(FIXED_ANGLES)

    buggy_novel, _ = render_grid(buggy_grid, W, H, FOV, novel_cam, novel_c2w, NEAR, FAR, N_SAMPLES)
    fixed_novel, _ = render_grid(fixed_grid, W, H, FOV, novel_cam, novel_c2w, NEAR, FAR, N_SAMPLES)
    buggy_novel_psnr = psnr(buggy_novel, novel_target)
    fixed_novel_psnr = psnr(fixed_novel, novel_target)

    return dict(
        buggy_hist=buggy_hist, fixed_hist=fixed_hist,
        buggy_train_psnr=buggy_train_psnr, fixed_train_psnr=fixed_train_psnr,
        buggy_novel_psnr=buggy_novel_psnr, fixed_novel_psnr=fixed_novel_psnr,
        novel_target=novel_target, buggy_novel=buggy_novel, fixed_novel=fixed_novel,
    )


if __name__ == "__main__":
    result = diagnose_and_fix()

    print(f"BUGGY  -- {len(BUGGY_ANGLES)} views {BUGGY_ANGLES} deg (a {BUGGY_ANGLES[1]-BUGGY_ANGLES[0]} deg baseline)")
    for it, mse in result["buggy_hist"]:
        print(f"  iter {it:4d}  train MSE = {mse:.5f}")
    print(f"  mean training-view PSNR = {result['buggy_train_psnr']:.2f} dB  (fits what it saw)")
    print(f"  held-out novel view ({NOVEL_ANGLE_DEG} deg) PSNR = {result['buggy_novel_psnr']:.2f} dB  <-- FLOATERS/COLLAPSE")

    print(f"\nFIXED  -- {len(FIXED_ANGLES)} views spread every 45 deg around the full ring")
    for it, mse in result["fixed_hist"]:
        print(f"  iter {it:4d}  train MSE = {mse:.5f}")
    print(f"  mean training-view PSNR = {result['fixed_train_psnr']:.2f} dB")
    print(f"  held-out novel view ({NOVEL_ANGLE_DEG} deg) PSNR = {result['fixed_novel_psnr']:.2f} dB")

    gap = result["fixed_novel_psnr"] - result["buggy_novel_psnr"]
    print(f"\nnovel-view PSNR improvement from spreading the views out: +{gap:.2f} dB")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    os.makedirs(out_dir, exist_ok=True)
    save_ppm(os.path.join(out_dir, "bug_novel_target.ppm"), result["novel_target"])
    save_ppm(os.path.join(out_dir, "bug_novel_buggy.ppm"), result["buggy_novel"])
    save_ppm(os.path.join(out_dir, "bug_novel_fixed.ppm"), result["fixed_novel"])
    print("wrote out/bug_novel_{target,buggy,fixed}.ppm")

    assert gap > 1.0, "expected the 8-view fix to clearly beat the 2-view bug on the held-out novel view"
    print("\n[ok] spreading the capture around the ring measurably fixes novel-view quality")
