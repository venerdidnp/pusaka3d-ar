"""Volume rendering: compositing samples along a ray into one pixel colour.
From scratch, NumPy only. This is the "z-buffer" of a NeRF -- instead of
keeping only the nearest opaque surface (Module 3), we accumulate a
weighted sum over every sample along the ray.
"""
from __future__ import annotations

import numpy as np


def alpha_from_density(sigma: float, delta: float) -> float:
    """The probability a sample of density sigma over a step of length delta
    absorbs/emits light: alpha = 1 - exp(-sigma * delta)."""
    return 1.0 - np.exp(-sigma * delta)


def composite(alphas, colors):
    """Front-to-back compositing. alphas: (N,) opacities already converted
    from density (or 1.0 for a hard, fully-opaque sample). colors: (N,3).
    Returns (final_color, weights, transmittance)."""
    alphas = np.asarray(alphas, dtype=float)
    colors = np.asarray(colors, dtype=float)
    n = len(alphas)
    transmittance = np.empty(n)
    weights = np.empty(n)
    t = 1.0
    for i in range(n):
        transmittance[i] = t
        weights[i] = t * alphas[i]
        t *= (1.0 - alphas[i])
    final_color = (weights[:, None] * colors).sum(axis=0)
    background_weight = t  # what's left over (unblocked background)
    return final_color, weights, transmittance, background_weight


def render_ray(sigmas, deltas, colors, is_opaque=None):
    """Convenience wrapper: sigmas/deltas -> alphas -> composite. A sample
    flagged in is_opaque (e.g. a hard surface hit) uses alpha=1 directly."""
    n = len(sigmas)
    if is_opaque is None:
        is_opaque = [False] * n
    alphas = [1.0 if op else alpha_from_density(s, d) for s, d, op in zip(sigmas, deltas, is_opaque)]
    return composite(alphas, colors)


def running_example():
    """Module 5 running example: 3 samples along a ray.
    Sample 1: sigma=1.0, delta=ln(2) ~ 0.6931  -> alpha = 0.5, colour red.
    Sample 2: sigma=1.0, delta=ln(2) ~ 0.6931  -> alpha = 0.5, colour green.
    Sample 3: an opaque surface hit             -> alpha = 1.0, colour blue.
    """
    ln2 = float(np.log(2.0))
    sigmas = [1.0, 1.0, None]
    deltas = [ln2, ln2, None]
    colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    is_opaque = [False, False, True]
    return render_ray(sigmas, deltas, colors, is_opaque)


if __name__ == "__main__":
    final_color, weights, transmittance, bg_weight = running_example()
    print("sample   T (before)   alpha    weight   colour")
    labels = ["1 (red)  ", "2 (green)", "3 (blue) "]
    alphas_shown = [0.5, 0.5, 1.0]
    for lab, t, a, w in zip(labels, transmittance, alphas_shown, weights):
        print(f"  {lab}   {t:.3f}         {a:.3f}    {w:.3f}")
    print(f"background weight (unblocked) = {bg_weight:.3f}")
    print(f"sum of weights + background   = {weights.sum() + bg_weight:.3f}  (must be 1.0)")
    print(f"final pixel colour (R,G,B)    = ({final_color[0]:.3f}, {final_color[1]:.3f}, {final_color[2]:.3f})")
