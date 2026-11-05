"""Capture and fit the opt-in temple-plus-sphere scene without changing defaults."""
from __future__ import annotations

import hashlib
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from capture import HERE, RING_ANGLES, build_capture, save_ppm
from fit_nerf_toy import LR, N_ITERS, NOVEL_ANGLE_DEG, RESOLUTION, load_capture
from nerf_toy import (
    EXTENDED_SPHERE, VoxelGrid, psnr, render_analytic, render_grid,
    ring_camera, scene_density_color, train,
)

ORIGINAL_SHA256 = "1b65c796393936248e86b2bd0fa67b6e99c04f0b99d1ff70fbcc02ba36a74d15"


def main():
    data_dir = os.path.join(HERE, "data", "pusaka_capture_extended")
    out_dir = os.path.join(HERE, "out")
    os.makedirs(out_dir, exist_ok=True)
    build_capture(data_dir=data_dir, extended=True)
    manifest, views, targets = load_capture(data_dir)
    assert manifest["extended"] is True and len(views) == 8
    W, H, FOV = manifest["width"], manifest["height"], manifest["fov_deg"]
    near, far, samples = manifest["near"], manifest["far"], manifest["n_samples"]
    args = (W, H, FOV)
    sampling = (near, far, samples)

    # Compare to the actual pre-edit renderer, not merely the new False branch.
    original_images = []
    fingerprint = hashlib.sha256()
    for angle in RING_ANGLES + [NOVEL_ANGLE_DEG]:
        camera = ring_camera(angle, manifest["radius"], manifest["camera_height"])
        image = render_analytic(*args, *camera, *sampling)
        original_images.append(image)
        fingerprint.update(image.tobytes())
        np.testing.assert_array_equal(
            image, render_analytic(*args, *camera, *sampling, extended=False))
    assert fingerprint.hexdigest() == ORIGINAL_SHA256, "Original scene changed"
    center = np.array([EXTENDED_SPHERE[0]])
    np.testing.assert_array_equal(scene_density_color(center)[0], [0.0])
    assert scene_density_color(center, extended=True)[0][0] > 0
    print("Original scene: exact pre-edit nine-view SHA256 match", fingerprint.hexdigest())
    print("Sphere (center, radius, RGB):", EXTENDED_SPHERE)
    print("Extended capture:", data_dir, "(8 views, extended=True manifest)")
    for frame in manifest["frames"]:
        print(f"  {frame['file']} angle={frame['angle_deg']} position={frame['position']}")
    assert any(not np.array_equal(a, b) for a, b in zip(targets, original_images))

    novel_camera = ring_camera(NOVEL_ANGLE_DEG, manifest["radius"], manifest["camera_height"])
    novel_target = render_analytic(*args, *novel_camera, *sampling, extended=True)
    grid = VoxelGrid(resolution=RESOLUTION, seed=0)

    def predictions():
        training = [render_grid(grid, *args, *camera, *sampling)[0] for camera in views]
        novel = render_grid(grid, *args, *novel_camera, *sampling)[0]
        return training, novel

    def save_images(stage, training, novel):
        for frame, target, pred in zip(manifest["frames"], targets, training):
            prefix = os.path.join(out_dir, f"extended_{stage}_{frame['angle_deg']:03d}")
            save_ppm(prefix + "_target.ppm", target)
            save_ppm(prefix + "_predicted.ppm", pred)
        prefix = os.path.join(out_dir, f"extended_{stage}_novel")
        save_ppm(prefix + "_target.ppm", novel_target)
        save_ppm(prefix + "_predicted.ppm", novel)
        print(f"Saved {stage} targets/predictions: out/extended_{stage}_*.ppm (18 files)")

    before_train, before_novel = predictions()
    save_images("before", before_train, before_novel)
    print(f"Before mean training-view PSNR = {np.mean([psnr(p, t) for p, t in zip(before_train, targets)]):.6f} dB")
    print(f"Before novel-view PSNR ({NOVEL_ANGLE_DEG} deg) = {psnr(before_novel, novel_target):.6f} dB")
    print(f"Training {RESOLUTION}^3 grid, seed=0, lr={LR}, {N_ITERS} iterations, "
          f"{W}x{H}, {samples} samples/ray")
    history = train(grid, views, *args, targets, *sampling,
                    lr=LR, n_iters=N_ITERS, log_every=200)
    for iteration, mse in history:
        print(f"  iter {iteration:4d} train MSE = {mse:.10f}")
    print("History:", history)
    after_train, after_novel = predictions()
    save_images("after", after_train, after_novel)
    scores = [psnr(pred, target) for pred, target in zip(after_train, targets)]
    for frame, score in zip(manifest["frames"], scores):
        print(f"After training view {frame['angle_deg']:3d}: PSNR = {score:.6f} dB")
    print(f"After mean training-view PSNR = {np.mean(scores):.6f} dB")
    print(f"After novel-view PSNR ({NOVEL_ANGLE_DEG} deg) = {psnr(after_novel, novel_target):.6f} dB")

    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for row, (target, before, after, label) in enumerate([
        (targets[0], before_train[0], after_train[0], "Train 0 deg"),
        (novel_target, before_novel, after_novel, "Novel 22.5 deg"),
    ]):
        for ax, image, title in zip(axes[row], [target, before, after],
                                    ["Target", "Before", "After"]):
            ax.imshow(np.clip(image, 0, 1), interpolation="nearest")
            ax.set_title(f"{label}: {title}")
            ax.axis("off")
    fig.tight_layout()
    comparison = os.path.join(out_dir, "extended_comparison.png")
    fig.savefig(comparison, dpi=160)
    plt.close(fig)
    print("Saved comparison:", comparison)

    # Extended rendering and training must not mutate the original scene.
    for angle, original in zip(RING_ANGLES + [NOVEL_ANGLE_DEG], original_images):
        camera = ring_camera(angle, manifest["radius"], manifest["camera_height"])
        np.testing.assert_array_equal(original, render_analytic(*args, *camera, *sampling))
    print("PASS: original scene numerically unchanged after extended capture and training (max difference 0).")


if __name__ == "__main__":
    main()
