"""Sweep raw density initialization using the original toy fitting settings."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fit_nerf_toy import LR, N_ITERS, RESOLUTION, load_capture
from nerf_toy import VoxelGrid, render_grid, train


def main():
    manifest, views, targets = load_capture()
    if len(views) != 8 or manifest.get("extended", False):
        raise ValueError("This sweep requires the eight original capture views.")
    width, height, fov = manifest["width"], manifest["height"], manifest["fov_deg"]
    near, far, n_samples = manifest["near"], manifest["far"], manifest["n_samples"]
    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(exist_ok=True)
    results = []
    fig, ax = plt.subplots(figsize=(9, 5.5))

    print(f"Original views={len(views)}, image={width}x{height}, samples/ray={n_samples}, "
          f"grid={RESOLUTION}^3, color seed=0, lr={LR}, updates={N_ITERS}, log_every=1",
          flush=True)
    print("MSE = mean squared error over all views, pixels, and RGB channels.", flush=True)
    print("History iteration i is pre-update (i updates completed); final is recomputed "
          f"after {N_ITERS} updates.", flush=True)
    print(f"{'init':>5} {'sigmoid(init)':>14} {'initial MSE':>15} "
          f"{'last pre MSE':>15} {'final post MSE':>15} {'train PSNR dB':>14}", flush=True)

    for init in (-5, -2, 0, 1):
        grid = VoxelGrid(resolution=RESOLUTION, seed=0)
        grid.sigma_raw.fill(init)
        history = train(grid, views, width, height, fov, targets, near, far, n_samples,
                        lr=LR, n_iters=N_ITERS, log_every=1)
        # Every history loss precedes its update, including the last entry.
        initial_mse = history[0][1]
        squared_error = 0.0
        n_values = 0
        for (cam_pos, c2w), target in zip(views, targets):
            prediction, _ = render_grid(grid, width, height, fov, cam_pos, c2w,
                                        near, far, n_samples)
            squared_error += float(np.sum((prediction - target) ** 2))
            n_values += target.size
        final_mse = squared_error / n_values
        derivative = float(VoxelGrid._sigmoid(float(init)))
        train_psnr = float(-10.0 * np.log10(final_mse))
        results.append({
            "sigma_raw_init": init,
            "softplus_derivative_at_init": derivative,
            "initial_mse": initial_mse,
            "last_pre_update_mse": history[-1][1],
            "final_post_update_mse": final_mse,
            "train_psnr_db_from_global_mse": train_psnr,
            "history_pre_update": history,
        })
        print(f"{init:5d} {derivative:14.10f} {initial_mse:15.10f} "
              f"{history[-1][1]:15.10f} {final_mse:15.10f} {train_psnr:14.6f}",
              flush=True)
        line, = ax.plot([it for it, _ in history], [mse for _, mse in history],
                        label=f"sigma_raw = {init}")
        ax.scatter([N_ITERS], [final_mse], color=line.get_color(), marker="x", s=40)

    ax.set(xlabel="Completed parameter updates (history is pre-update)",
           ylabel="Train MSE (all views, pixels, RGB)",
           title="Raw density initialization sweep: original 8 views")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(title=f"Lines: history 0-{N_ITERS - 1}; x: post-update {N_ITERS}")
    fig.tight_layout()
    plot_path = out_dir / "sigma_init_sweep.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    history_path = out_dir / "sigma_init_sweep.json"
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump({
            "settings": {
                "resolution": RESOLUTION, "color_seed": 0, "lr": LR,
                "n_iters": N_ITERS, "log_every": 1, "n_views": len(views),
                "width": width, "height": height, "fov_deg": fov,
                "near": near, "far": far, "n_samples": n_samples,
                "mse_reduction": "mean over all views, pixels, RGB channels",
                "history_timing": "pre-update; iteration i has i completed updates",
                "final_mse_timing": "recomputed after all updates",
            },
            "results": results,
        }, handle, indent=2, allow_nan=False)
    print("Files: out/sigma_init_sweep.png, out/sigma_init_sweep.json", flush=True)


if __name__ == "__main__":
    main()
