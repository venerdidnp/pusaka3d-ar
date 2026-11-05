"""A from-scratch, toy differentiable volume renderer + gradient-descent
fitter -- the core NeRF idea (reconstruct a 3D scene from 2D multi-view
images by making rendering differentiable) at a scale small enough to run
on a CPU in seconds and to unit-test against finite differences.

No PyTorch / JAX / autograd library anywhere: the backward pass below is
hand-derived analytically (see code/Module5/README.md for the derivation)
and verified against numerical (finite-difference) gradients in tests.py.
This is *in addition to* the real tool (nerfstudio / Instant-NGP), described
and commanded in Module5_EN.tex's Software section -- see the "Known
deviations" note in README.md for why full GPU training is not run here.

Scene: a small stepped "candi" (temple) silhouette, analytic ground truth,
inside the cube [-1,1]^3. Reconstructed as a voxel grid of (density, colour)
via nearest-voxel lookup (not trilinear -- see README) so the backward pass
is a simple scatter-add.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Ground-truth analytic scene (the "real world" the synthetic cameras see)
# ---------------------------------------------------------------------------

_TIERS = [
    # (half_x, half_z, y_lo, y_hi, color)
    (0.90, 0.90, -1.00, -0.60, (0.55, 0.50, 0.45)),
    (0.65, 0.65, -0.60, -0.20, (0.50, 0.45, 0.40)),
    (0.40, 0.40, -0.20, 0.30, (0.45, 0.40, 0.38)),
    (0.15, 0.15, 0.30, 0.80, (0.78, 0.64, 0.20)),
]
SCENE_SIGMA = 15.0  # opaque enough that a handful of samples suffices


EXTENDED_SPHERE = ((0.72, 0.12, 0.0), 0.24, (0.12, 0.65, 0.85))


def scene_density_color(points: np.ndarray, extended=False):
    """points: (..., 3). Returns (sigma, color) arrays, sigma: (...,),
    color: (..., 3). Analytic ground truth -- not a grid."""
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    sigma = np.zeros(points.shape[:-1])
    color = np.zeros(points.shape[:-1] + (3,))
    for hx, hz, ylo, yhi, c in _TIERS:
        inside = (np.abs(x) <= hx) & (np.abs(z) <= hz) & (y >= ylo) & (y <= yhi)
        sigma = np.where(inside, SCENE_SIGMA, sigma)
        for k in range(3):
            color[..., k] = np.where(inside, c[k], color[..., k])
    if extended:
        center, radius, sphere_color = EXTENDED_SPHERE
        inside = np.sum((points - np.array(center)) ** 2, axis=-1) <= radius ** 2
        sigma = np.where(inside, SCENE_SIGMA, sigma)
        color = np.where(inside[..., None], sphere_color, color)
    return sigma, color


# ---------------------------------------------------------------------------
# Cameras: a minimal look-at, reused from Module 3's idea, self-contained here
# ---------------------------------------------------------------------------

def look_at_c2w(eye: np.ndarray, target: np.ndarray, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    """3x3 camera-to-world rotation whose columns are the camera's world-space
    +x, +y, +z axes (the OpenGL convention: camera looks down its own -z)."""
    up = np.array(up, dtype=float)
    forward = np.array(target, dtype=float) - np.array(eye, dtype=float)
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)
    # camera's -z axis is `forward` (looks down -z), +x is `right`, +y is true_up
    return np.stack([right, true_up, -forward], axis=1)


def ring_camera(angle_deg: float, radius: float = 3.0, height: float = 1.0):
    ang = np.radians(angle_deg)
    eye = np.array([radius * np.cos(ang), height, radius * np.sin(ang)])
    c2w = look_at_c2w(eye, np.array([0.0, -0.1, 0.0]))
    return eye, c2w


# ---------------------------------------------------------------------------
# Ray generation (reuses ray_generator.py's convention) + sampling
# ---------------------------------------------------------------------------

def generate_rays(width, height, fov_deg, cam_pos, c2w):
    f = 0.5 * width / np.tan(np.radians(fov_deg) / 2.0)
    cx, cy = width / 2.0, height / 2.0
    us, vs = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    x = (us - cx) / f
    y = -(vs - cy) / f
    z = -np.ones_like(x)
    dirs_cam = np.stack([x, y, z], axis=-1)
    dirs_world = dirs_cam @ c2w.T
    dirs_world = dirs_world / np.linalg.norm(dirs_world, axis=-1, keepdims=True)
    origins = np.broadcast_to(cam_pos, dirs_world.shape)
    return origins, dirs_world


def sample_points(origins, dirs, near, far, n_samples):
    """Stratified-free (fixed) sample depths for reproducibility."""
    t = np.linspace(near, far, n_samples)
    delta = t[1] - t[0]
    pts = origins[..., None, :] + dirs[..., None, :] * t[None, None, :, None] \
        if origins.ndim == 3 else origins[:, None, :] + dirs[:, None, :] * t[None, :, None]
    return pts, t, delta


# ---------------------------------------------------------------------------
# Analytic-scene rendering (produces the "captured" training/test images)
# ---------------------------------------------------------------------------

def render_analytic(width, height, fov_deg, cam_pos, c2w, near=1.0, far=5.0, n_samples=32,
                     background=(1.0, 1.0, 1.0), extended=False):
    origins, dirs = generate_rays(width, height, fov_deg, cam_pos, c2w)
    pts, t, delta = sample_points(origins, dirs, near, far, n_samples)
    sigma, color = scene_density_color(pts, extended=extended)
    alpha = 1.0 - np.exp(-sigma * delta)
    image, _weights, _bg_weight = _composite_image(alpha, color, background)
    return image


def _composite_image(alpha, color, background):
    """alpha: (H,W,N), color: (H,W,N,3). Vectorised front-to-back compositing."""
    one_minus = 1.0 - alpha
    # transmittance before sample i = cumulative product of (1-alpha) for j<i
    T = np.cumprod(np.concatenate([np.ones_like(one_minus[..., :1]), one_minus[..., :-1]], axis=-1), axis=-1)
    weights = T * alpha
    image = (weights[..., None] * color).sum(axis=-2)
    bg_weight = T[..., -1] * one_minus[..., -1]
    image = image + bg_weight[..., None] * np.array(background)
    return image, weights, bg_weight


def psnr(a, b, data_range=1.0):
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 10.0 * np.log10((data_range ** 2) / mse)


# ---------------------------------------------------------------------------
# The learnable voxel grid (nearest-neighbour lookup -> trivial backward)
# ---------------------------------------------------------------------------

class VoxelGrid:
    def __init__(self, resolution=10, extent=1.0, seed=0):
        rng = np.random.default_rng(seed)
        self.res = resolution
        self.extent = extent
        # raw (pre-activation) parameters. sigma_raw starts at 0 (softplus(0)
        # = ln2, sigmoid(0) = 0.5) deliberately -- a very negative start
        # (an "empty" scene) crushes the gradient through softplus'(x) =
        # sigmoid(x) to near zero, and training never escapes it. Starting
        # neutral keeps the gradient signal alive from step 0 (see
        # README.md "Notes on fixes applied while verifying").
        self.sigma_raw = np.zeros((resolution, resolution, resolution))
        self.color_raw = rng.normal(0.0, 0.01, size=(resolution, resolution, resolution, 3))
        self.sigma_grad = np.zeros_like(self.sigma_raw)
        self.color_grad = np.zeros_like(self.color_raw)

    def _voxel_index(self, points):
        # map [-extent, extent]^3 -> [0, res-1]^3 integer index, clamped
        idx = np.floor((points + self.extent) / (2 * self.extent) * self.res).astype(int)
        return np.clip(idx, 0, self.res - 1)

    @staticmethod
    def _softplus(x):
        return np.logaddexp(0.0, x)

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, points):
        """points: (...,3). Returns sigma (...), color (...,3), and the
        flat voxel indices (...) used, for the backward pass."""
        idx = self._voxel_index(points)
        ix, iy, iz = idx[..., 0], idx[..., 1], idx[..., 2]
        sigma_raw = self.sigma_raw[ix, iy, iz]
        color_raw = self.color_raw[ix, iy, iz]
        sigma = self._softplus(sigma_raw)
        color = self._sigmoid(color_raw)
        flat = (ix * self.res + iy) * self.res + iz
        return sigma, color, flat, sigma_raw

    def zero_grad(self):
        self.sigma_grad.fill(0.0)
        self.color_grad.fill(0.0)

    def accumulate(self, flat_idx, d_sigma_raw, d_color_raw):
        """Scatter-add gradients w.r.t. raw params into the grid's grad buffers."""
        np.add.at(self.sigma_grad.reshape(-1), flat_idx, d_sigma_raw)
        cg = self.color_grad.reshape(-1, 3)
        np.add.at(cg, flat_idx, d_color_raw)

    def step(self, lr):
        self.sigma_raw -= lr * self.sigma_grad
        self.color_raw -= lr * self.color_grad


def render_grid(grid: VoxelGrid, width, height, fov_deg, cam_pos, c2w,
                 near=1.0, far=5.0, n_samples=32, background=(1.0, 1.0, 1.0)):
    """Forward pass through the learnable grid. Returns everything backward()
    needs."""
    origins, dirs = generate_rays(width, height, fov_deg, cam_pos, c2w)
    pts, t, delta = sample_points(origins, dirs, near, far, n_samples)
    flat_shape = pts.shape[:-1]  # (H, W, N)
    sigma, color, flat_idx, sigma_raw = grid.forward(pts.reshape(-1, 3))
    sigma = sigma.reshape(flat_shape)
    color = color.reshape(flat_shape + (3,))
    sigma_raw = sigma_raw.reshape(flat_shape)
    flat_idx = flat_idx.reshape(flat_shape)

    alpha = 1.0 - np.exp(-sigma * delta)
    image, weights, bg_weight = _composite_image(alpha, color, background)

    cache = dict(alpha=alpha, color=color, weights=weights, bg_weight=bg_weight,
                 sigma_raw=sigma_raw, flat_idx=flat_idx, delta=delta, background=background)
    return image, cache


def backward_grid(grid: VoxelGrid, cache, d_image):
    """Analytic backward pass for one rendered image.

    d_image: (H,W,3) upstream gradient dLoss/dImage.

    Derivation (per ray, N samples, alpha_i, colour c_i, weight w_i = T_i*alpha_i):
        dC/dc_i     = w_i
        dC/dalpha_i = T_i*c_i - (suffix_i + w_bg*c_bg) / (1 - alpha_i)
    where suffix_i = sum_{j>i} w_j*c_j (a reverse cumulative sum), and the
    division is skipped (contributes 0) when alpha_i == 1 exactly.
    """
    alpha = cache["alpha"]                     # (H,W,N)
    color = cache["color"]                     # (H,W,N,3)
    weights = cache["weights"]                 # (H,W,N)
    bg_weight = cache["bg_weight"]              # (H,W)
    background = np.array(cache["background"])  # (3,)
    delta = cache["delta"]
    sigma_raw = cache["sigma_raw"]              # (H,W,N)
    flat_idx = cache["flat_idx"]                # (H,W,N)

    one_minus = 1.0 - alpha
    T = weights / np.clip(alpha, 1e-8, None)    # T_i (recovered; safe since alpha>=0)

    # dC/dc_i (per channel) = w_i ; fold in upstream d_image via chain rule
    d_color = weights[..., None] * d_image[..., None, :]   # (H,W,N,3)

    # suffix_i = sum_{j>i} w_j * (c_j . d_image)   -- reduce colour dim with d_image first
    wc_dot = (weights[..., None] * color * d_image[..., None, :]).sum(axis=-1)  # (H,W,N)
    suffix = np.cumsum(wc_dot[..., ::-1], axis=-1)[..., ::-1] - wc_dot          # sum_{j>i}
    bg_dot = bg_weight * (background * d_image).sum(axis=-1)                   # w_bg*(c_bg . d_image)

    denom = np.clip(one_minus, 1e-6, None)
    d_alpha_direct = T * (color * d_image[..., None, :]).sum(axis=-1)
    d_alpha = d_alpha_direct - (suffix + bg_dot[..., None]) / denom
    # where alpha == 1 (opaque sample), the (suffix+bg)/(1-alpha) term is defined to be 0
    d_alpha = np.where(alpha >= 1.0 - 1e-9, d_alpha_direct, d_alpha)

    # chain through alpha = 1 - exp(-sigma*delta), so d(alpha)/d(sigma) = delta*(1-alpha)
    d_sigma = d_alpha * delta * (1.0 - alpha)
    d_sigma_raw = d_sigma * VoxelGrid._sigmoid(sigma_raw)  # softplus'(x) = sigmoid(x)

    d_color_raw = d_color * color[..., :] * (1.0 - color[..., :])  # sigmoid'(x) = s(1-s)

    grid.accumulate(flat_idx.reshape(-1), d_sigma_raw.reshape(-1), d_color_raw.reshape(-1, 3))


def train(grid: VoxelGrid, views, width, height, fov_deg, targets,
          near=1.0, far=5.0, n_samples=32, lr=5.0, n_iters=300, log_every=50):
    history = []
    n_pix = width * height * len(views)
    for it in range(n_iters):
        grid.zero_grad()
        total_loss = 0.0
        for (cam_pos, c2w), target in zip(views, targets):
            image, cache = render_grid(grid, width, height, fov_deg, cam_pos, c2w, near, far, n_samples)
            diff = image - target
            total_loss += float(np.sum(diff ** 2))
            d_image = 2.0 * diff / n_pix
            backward_grid(grid, cache, d_image)
        grid.step(lr)
        mse = total_loss / (n_pix * 3)
        if it % log_every == 0 or it == n_iters - 1:
            history.append((it, mse))
    return history
