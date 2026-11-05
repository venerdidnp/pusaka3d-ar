"""Generate the committed, deterministic "capture" dataset -- 8 synthetic
posed views of the toy Pusaka3D temple, ring-arranged like a COLMAP capture.
No RNG, no binary photo assets: everything here is produced by
nerf_toy.py's analytic scene, so it is exactly reproducible from source
(MASTER Sec 9.2's "deterministic input" rule). A real phone capture of the
student's own subject, run through nerfstudio/COLMAP, is the graded,
bring-your-own-input version this stands in for.
"""
from __future__ import annotations

import json
import os

import numpy as np

from nerf_toy import EXTENDED_SPHERE, render_analytic, ring_camera

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "pusaka_capture")

WIDTH, HEIGHT, FOV_DEG = 20, 16, 50.0
NEAR, FAR, N_SAMPLES = 1.0, 5.0, 32
RING_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]
RADIUS, HEIGHT_CAM = 3.0, 1.0


def save_ppm(path, image):
    """8-bit binary PPM (P6), the same format code/Module1 and code/Module3
    already use -- one dependency-free image format across the whole
    course."""
    img8 = np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8)
    h, w, _ = img8.shape
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        f.write(img8.tobytes())


def build_capture(data_dir=DATA_DIR, extended=False):
    os.makedirs(data_dir, exist_ok=True)
    poses = []
    for angle in RING_ANGLES:
        cam_pos, c2w = ring_camera(angle, radius=RADIUS, height=HEIGHT_CAM)
        image = render_analytic(WIDTH, HEIGHT, FOV_DEG, cam_pos, c2w, NEAR, FAR, N_SAMPLES,
                                extended=extended)
        fname = f"view_{angle:03d}.ppm"
        save_ppm(os.path.join(data_dir, fname), image)
        poses.append({
            "file": fname,
            "angle_deg": angle,
            "position": cam_pos.tolist(),
            "camera_to_world_rotation": c2w.tolist(),
        })

    manifest = {
        "width": WIDTH, "height": HEIGHT, "fov_deg": FOV_DEG,
        "near": NEAR, "far": FAR, "n_samples": N_SAMPLES,
        "radius": RADIUS, "camera_height": HEIGHT_CAM,
        "frames": poses,
        "_comment": "Deterministic synthetic ring capture of the toy Pusaka3D "
                    "temple scene defined in nerf_toy.py -- COLMAP-style pose "
                    "manifest, no RNG, no binary photo assets.",
    }
    if extended:
        center, radius, color = EXTENDED_SPHERE
        manifest["extended"] = True
        manifest["sphere"] = {"center": center, "radius": radius, "color": color}
        manifest["_comment"] = (
            "Deterministic synthetic ring capture of the toy Pusaka3D temple "
            "plus one sphere, rendered with extended=True in nerf_toy.py."
        )
    with open(os.path.join(data_dir, "poses.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


if __name__ == "__main__":
    manifest = build_capture()
    print(f"wrote {len(manifest['frames'])} views to {DATA_DIR}")
    for frame in manifest["frames"]:
        print(f"  {frame['file']}  angle={frame['angle_deg']:>3} deg  pos={[round(c,3) for c in frame['position']]}")
