"""Module 5 test suite. Run: python tests.py -> ALL CHECKS PASSED (N ok)

Includes a finite-difference check of the hand-derived analytic backward
pass (nerf_toy.backward_grid) -- the from-scratch "match a reference"
moment applied to the from-scratch code itself, since there is no external
autodiff library in play to check against.
"""
from __future__ import annotations

import argparse

import numpy as np

from ray_generator import running_example as ray_running_example
from volume_render import running_example as vr_running_example
from nerf_toy import VoxelGrid, render_grid, backward_grid, ring_camera
import nerf_bug

_checks = 0


def check(name, got, want, tol=1e-6):
    ok = abs(got - want) <= tol
    status = "ok" if ok else "FAIL"
    print(f"[{status}] {name}: got={got!r} want={want!r}")
    assert ok, f"{name}: got {got!r}, want {want!r}"
    global _checks
    _checks += 1


def check_vec(name, got, want, tol=1e-3):
    for g, w in zip(got, want):
        check(f"{name} ({w})", g, w, tol)


def test_ray_generator():
    dir_cam, norm_dir = ray_running_example()
    check_vec("E1 pinhole ray, camera-space (unnormalised)", dir_cam, (0.8, 0.0, -1.0))
    check_vec("E1 pinhole ray, normalised", norm_dir, (0.625, 0.0, -0.781), tol=1e-3)
    check("E1 normalised ray has unit length", float(np.linalg.norm(norm_dir)), 1.0, tol=1e-9)


def test_volume_render():
    final_color, weights, transmittance, bg_weight = vr_running_example()
    check_vec("E2 volume-render weights", weights, (0.5, 0.25, 0.25))
    check("E2 background weight", bg_weight, 0.0, tol=1e-9)
    check("E2 weights + background sum to 1", float(weights.sum() + bg_weight), 1.0, tol=1e-9)
    check_vec("E2 final pixel colour", final_color, (0.5, 0.25, 0.25))


def test_backward_matches_finite_difference(break_on_purpose=False):
    grid = VoxelGrid(resolution=4, seed=1)
    rng = np.random.default_rng(2)
    grid.sigma_raw = rng.normal(0, 1, size=grid.sigma_raw.shape)
    grid.color_raw = rng.normal(0, 1, size=grid.color_raw.shape)

    W, H, FOV = 4, 3, 50.0
    cam_pos, c2w = ring_camera(30.0, radius=3.0, height=1.0)
    target = np.random.default_rng(4).uniform(0, 1, size=(H, W, 3))

    def loss_and_render():
        image, cache = render_grid(grid, W, H, FOV, cam_pos, c2w, near=1.0, far=5.0, n_samples=8)
        return float(np.sum((image - target) ** 2)), image, cache

    _, image0, cache = loss_and_render()
    d_image = 2.0 * (image0 - target)
    grid.zero_grad()
    backward_grid(grid, cache, d_image)
    analytic_sigma_grad = grid.sigma_grad.copy()
    analytic_color_grad = grid.color_grad.copy()

    eps = 1e-4
    max_err = 0.0
    worst = None
    rng2 = np.random.default_rng(5)
    probe_idxs = [tuple(rng2.integers(0, 4, size=3)) for _ in range(6)]
    if break_on_purpose:
        nonzero_idx = np.unravel_index(np.argmax(np.abs(analytic_sigma_grad)), analytic_sigma_grad.shape)
        assert abs(analytic_sigma_grad[nonzero_idx]) > 1e-6, "need a nonzero density gradient to corrupt"
        probe_idxs.append(nonzero_idx)
        analytic_sigma_grad *= -1.0

    for idx in probe_idxs:
        orig = grid.sigma_raw[idx]
        grid.sigma_raw[idx] = orig + eps
        lp, _, _ = loss_and_render()
        grid.sigma_raw[idx] = orig - eps
        lm, _, _ = loss_and_render()
        grid.sigma_raw[idx] = orig
        numeric = (lp - lm) / (2 * eps)
        analytic = float(analytic_sigma_grad[idx])
        err = abs(numeric - analytic)
        if worst is None or err > max_err:
            max_err = err
            worst = ("sigma_raw", tuple(int(i) for i in idx), analytic, numeric)

    for idx in probe_idxs:
        for ch in range(3):
            full_idx = idx + (ch,)
            orig = grid.color_raw[full_idx]
            grid.color_raw[full_idx] = orig + eps
            lp, _, _ = loss_and_render()
            grid.color_raw[full_idx] = orig - eps
            lm, _, _ = loss_and_render()
            grid.color_raw[full_idx] = orig
            numeric = (lp - lm) / (2 * eps)
            analytic = float(analytic_color_grad[full_idx])
            err = abs(numeric - analytic)
            if worst is None or err > max_err:
                max_err = err
                worst = ("color_raw", tuple(int(i) for i in full_idx), analytic, numeric)

    print(f"[{'ok' if max_err < 1e-6 else 'FAIL'}] analytic vs finite-difference gradient, max|err| = {max_err:.17g}")
    parameter, idx, analytic, numeric = worst
    print(f"worst parameter: {parameter}{idx}, analytic={analytic:.17g}, numeric={numeric:.17g}")
    assert max_err < 1e-6, f"analytic gradient does not match finite differences: max err {max_err:.17g}"
    global _checks
    _checks += 1


def test_view_coverage_diagnose_and_fix():
    result = nerf_bug.diagnose_and_fix()
    gap = result["fixed_novel_psnr"] - result["buggy_novel_psnr"]
    print(f"[{'ok' if gap > 1.0 else 'FAIL'}] 8-view vs 2-view novel-view PSNR gap = {gap:.2f} dB (want > 1.0 dB)")
    assert gap > 1.0, f"expected spreading the views to help novel-view PSNR by >1 dB, got {gap:.2f} dB"
    global _checks
    _checks += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--break-on-purpose", action="store_true",
                        help="deliberately corrupt the finite-difference test's analytic density gradient")
    args = parser.parse_args()
    test_ray_generator()
    test_volume_render()
    test_backward_matches_finite_difference(break_on_purpose=args.break_on_purpose)
    test_view_coverage_diagnose_and_fix()
    print(f"\nALL CHECKS PASSED ({_checks} ok)")
