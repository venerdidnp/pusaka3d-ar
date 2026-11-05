"""The pinhole camera ray generator -- from scratch, NumPy only.

Inverts Module 3's perspective projection: instead of taking a 3D point and
projecting it to a pixel, we take a pixel and generate the 3D ray (origin +
direction) that would have projected onto it. This is the camera model
behind every NeRF pipeline.
"""
from __future__ import annotations

import numpy as np


def pixel_to_camera_dir(u: float, v: float, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Direction (in *camera* space, not yet normalised) of the ray through
    pixel (u, v), for a camera looking down -z with +x right, +y up (the
    OpenGL / Module 3 convention). Image v grows downward, so the y term is
    negated to flip into the up-is-positive camera frame.
    """
    x = (u - cx) / fx
    y = -(v - cy) / fy
    z = -1.0
    return np.array([x, y, z])


def camera_to_world_ray(origin_dir_cam: np.ndarray, cam_to_world: np.ndarray, cam_pos: np.ndarray):
    """Rotate a camera-space direction into world space and pair it with the
    camera's world position as the ray origin."""
    direction_world = cam_to_world @ origin_dir_cam
    return cam_pos.copy(), direction_world


def generate_rays(width: int, height: int, fx: float, fy: float, cx: float, cy: float,
                   cam_to_world: np.ndarray, cam_pos: np.ndarray):
    """Return (origins, directions) arrays of shape (H, W, 3) -- one ray per
    pixel, directions normalised."""
    us, vs = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    x = (us - cx) / fx
    y = -(vs - cy) / fy
    z = -np.ones_like(x)
    dirs_cam = np.stack([x, y, z], axis=-1)                  # (H, W, 3)
    dirs_world = dirs_cam @ cam_to_world.T                   # rotate each direction
    dirs_world = dirs_world / np.linalg.norm(dirs_world, axis=-1, keepdims=True)
    origins = np.broadcast_to(cam_pos, dirs_world.shape)
    return origins, dirs_world


def running_example():
    """Module 5 running example: f=100 px, principal point (320,240),
    pixel (400,240), identity camera pose."""
    fx = fy = 100.0
    cx, cy = 320.0, 240.0
    u, v = 400.0, 240.0
    dir_cam = pixel_to_camera_dir(u, v, fx, fy, cx, cy)
    norm_dir = dir_cam / np.linalg.norm(dir_cam)
    return dir_cam, norm_dir


if __name__ == "__main__":
    dir_cam, norm_dir = running_example()
    print(f"pixel (400, 240), f=100, principal point (320, 240)")
    print(f"unnormalised camera-space direction = ({dir_cam[0]:.3f}, {dir_cam[1]:.3f}, {dir_cam[2]:.3f})")
    print(f"normalised direction                = ({norm_dir[0]:.3f}, {norm_dir[1]:.3f}, {norm_dir[2]:.3f})")
