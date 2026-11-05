"""Build pusaka_temple.glb: a small, procedurally-generated, low-poly candi
(temple) mesh -- the AR app's model. Deterministic (no RNG), built entirely
from primitive boxes/pyramids with per-face flat normals, one glTF material
per part so the tiers, roof, and finial are visually distinct without a
texture image. Also writes hotspots.json: the 3D position/normal of each
named architectural part, with bilingual (EN/ID) educational text, for the
AR app's tap-to-learn hotspots.

Run: python make_temple_glb.py   -> writes pusaka_temple.glb + hotspots.json
next to this script (ar_app/assets/).
"""
from __future__ import annotations

import json
import os
import struct

import numpy as np
import pygltflib as gltf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "assets")


# ---------------------------------------------------------------------------
# Geometry: boxes and an n-sided pyramid, with flat (per-face, unshared) normals
# ---------------------------------------------------------------------------

def box(center, half_extents):
    """Return (positions, normals, indices) for an axis-aligned box, 24
    vertices (4 per face, unshared) so normals are flat per face."""
    cx, cy, cz = center
    hx, hy, hz = half_extents
    # 6 faces, each defined by 4 corners (CCW when viewed from outside) + its normal
    faces = [
        ((-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz), (0, 0, 1)),   # +z
        ((hx, -hy, -hz), (-hx, -hy, -hz), (-hx, hy, -hz), (hx, hy, -hz), (0, 0, -1)),  # -z
        ((hx, -hy, hz), (hx, -hy, -hz), (hx, hy, -hz), (hx, hy, hz), (1, 0, 0)),   # +x
        ((-hx, -hy, -hz), (-hx, -hy, hz), (-hx, hy, hz), (-hx, hy, -hz), (-1, 0, 0)),  # -x
        ((-hx, hy, hz), (hx, hy, hz), (hx, hy, -hz), (-hx, hy, -hz), (0, 1, 0)),   # +y
        ((-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz), (-hx, -hy, hz), (0, -1, 0)),  # -y
    ]
    positions, normals, indices = [], [], []
    for a, b, c, d, n in faces:
        base = len(positions)
        for p in (a, b, c, d):
            positions.append((p[0] + cx, p[1] + cy, p[2] + cz))
            normals.append(n)
        indices += [base, base + 1, base + 2, base, base + 2, base + 3]
    return np.array(positions, dtype=np.float32), np.array(normals, dtype=np.float32), np.array(indices, dtype=np.uint16)


def pyramid(center, base_half, height, sides=4):
    """An n-sided pyramid (default 4 = a square spire), apex pointing +y,
    base at y=center_y."""
    cx, cy, cz = center
    apex = np.array([cx, cy + height, cz], dtype=np.float32)
    angles = np.linspace(0, 2 * np.pi, sides, endpoint=False) + np.pi / sides
    base_pts = [np.array([cx + base_half * np.cos(a), cy, cz + base_half * np.sin(a)], dtype=np.float32) for a in angles]

    positions, normals, indices = [], [], []
    # side faces
    for i in range(sides):
        p0, p1 = base_pts[i], base_pts[(i + 1) % sides]
        edge1, edge2 = p1 - p0, apex - p0
        n = np.cross(edge1, edge2)
        n = n / (np.linalg.norm(n) + 1e-9)
        base_idx = len(positions)
        for p in (p0, p1, apex):
            positions.append(tuple(p))
            normals.append(tuple(n))
        indices += [base_idx, base_idx + 1, base_idx + 2]
    # base face (pointing down)
    base_idx = len(positions)
    for p in reversed(base_pts):
        positions.append(tuple(p))
        normals.append((0.0, -1.0, 0.0))
    for i in range(1, sides - 1):
        indices += [base_idx, base_idx + i, base_idx + i + 1]
    return np.array(positions, dtype=np.float32), np.array(normals, dtype=np.float32), np.array(indices, dtype=np.uint16)


# ---------------------------------------------------------------------------
# The candi: 4 tiers (cosmology: kaki / tubuh / tubuh atas / atap), a relief
# panel, and a stupa finial. Metres; sits on a table at a plausible ~46 cm.
# ---------------------------------------------------------------------------

PARTS = {
    # name: (kind, args, color RGBA, part id, y-range for hotspot placement)
    "base":   dict(kind="box", center=(0, 0.04, 0), half=(0.12, 0.04, 0.12), color=(0.55, 0.50, 0.45, 1.0)),
    "body":   dict(kind="box", center=(0, 0.12, 0), half=(0.09, 0.04, 0.09), color=(0.50, 0.45, 0.40, 1.0)),
    "upper":  dict(kind="box", center=(0, 0.20, 0), half=(0.06, 0.04, 0.06), color=(0.46, 0.42, 0.39, 1.0)),
    "roof":   dict(kind="box", center=(0, 0.27, 0), half=(0.045, 0.03, 0.045), color=(0.40, 0.36, 0.34, 1.0)),
    "relief": dict(kind="box", center=(0, 0.12, 0.091), half=(0.05, 0.025, 0.003), color=(0.35, 0.30, 0.22, 1.0)),
    "stupa":  dict(kind="pyramid", center=(0, 0.30, 0), base_half=0.03, height=0.12, sides=4,
                   color=(0.80, 0.66, 0.22, 1.0)),
}

HOTSPOTS = [
    dict(id="base", part="base", position=(0.12, 0.04, 0.12), normal=(0.71, 0.0, 0.71),
         name_en="Base (kaki candi)", name_id="Kaki Candi",
         text_en="The lowest tier represents Bhurloka, the world of ordinary "
                 "human life, in the three-part cosmology many Hindu-Buddhist "
                 "candi in Java and Bali follow. It is also the structural "
                 "foundation carrying the whole building's weight.",
         text_id="Tingkat terbawah melambangkan Bhurloka, alam kehidupan "
                  "manusia biasa, dalam kosmologi tiga tingkat yang dianut "
                  "banyak candi Hindu-Buddha di Jawa dan Bali. Bagian ini "
                  "juga menjadi fondasi struktural yang menopang seluruh "
                  "bangunan."),
    dict(id="body", part="body", position=(0.09, 0.12, 0.09), normal=(0.71, 0.0, 0.71),
         name_en="Body (tubuh candi)", name_id="Tubuh Candi",
         text_en="The body represents Bhuvarloka, the middle realm. In a "
                 "real candi this section usually holds the inner chamber "
                 "(garbhagriha) where a statue of a deity was placed.",
         text_id="Tubuh candi melambangkan Bhuvarloka, alam pertengahan. "
                  "Pada candi sesungguhnya, bagian ini biasanya menyimpan "
                  "bilik utama (garbhagriha) tempat arca dewa diletakkan."),
    dict(id="relief", part="relief", position=(0.05, 0.12, 0.094), normal=(0.0, 0.0, 1.0),
         name_en="Relief panel", name_id="Panel Relief",
         text_en="Real candi walls are often carved with narrative reliefs -- "
                 "story panels meant to be 'read' by walking around the "
                 "building, teaching religious epics to visitors who could "
                 "not read.",
         text_id="Dinding candi asli sering dipahat relief naratif -- panel "
                  "cerita yang dimaksudkan untuk 'dibaca' sambil mengelilingi "
                  "bangunan, mengajarkan kisah keagamaan kepada pengunjung "
                  "yang tidak bisa membaca."),
    dict(id="roof", part="roof", position=(0.045, 0.27, 0.045), normal=(0.71, 0.0, 0.71),
         name_en="Roof (atap)", name_id="Atap",
         text_en="The tiered roof represents Svarloka, the heavens. Each "
                 "receding tier steps the building visually toward the sky, "
                 "toward the finial at its peak.",
         text_id="Atap bertingkat melambangkan Svarloka, alam para dewa. "
                  "Setiap tingkat yang mengecil membawa bangunan secara "
                  "visual naik menuju langit, menuju puncak di ujungnya."),
    dict(id="stupa", part="stupa", position=(0.0, 0.42, 0.0), normal=(0.0, 1.0, 0.0),
         name_en="Finial (stupa / puncak)", name_id="Puncak / Stupa",
         text_en="The finial crowns the temple and, in Hindu-Buddhist "
                 "cosmology, represents the peak of Mount Meru -- the "
                 "mythical cosmic mountain the entire candi is a model of.",
         text_id="Puncak menahkotai candi dan, dalam kosmologi Hindu-Buddha, "
                  "melambangkan puncak Gunung Meru -- gunung kosmis mitologis "
                  "yang direpresentasikan oleh seluruh bangunan candi."),
]


def build_mesh_arrays():
    all_pos, all_norm, all_idx = [], [], []
    primitives = []  # (index_start, index_count, color)
    for name, spec in PARTS.items():
        if spec["kind"] == "box":
            pos, norm, idx = box(spec["center"], spec["half"])
        else:
            pos, norm, idx = pyramid(spec["center"], spec["base_half"], spec["height"], spec["sides"])
        vert_offset = len(all_pos)
        all_pos.append(pos)
        all_norm.append(norm)
        all_idx.append(idx + vert_offset)
        primitives.append((name, len(all_idx[-1]), spec["color"], idx.max() + 1 + vert_offset - vert_offset))
    return all_pos, all_norm, all_idx, primitives


def write_glb(path):
    all_pos, all_norm, all_idx, part_names = [], [], [], []
    positions_list, normals_list, indices_list, colors = [], [], [], []
    for name, spec in PARTS.items():
        if spec["kind"] == "box":
            pos, norm, idx = box(spec["center"], spec["half"])
        else:
            pos, norm, idx = pyramid(spec["center"], spec["base_half"], spec["height"], spec["sides"])
        positions_list.append(pos)
        normals_list.append(norm)
        indices_list.append(idx)
        colors.append(spec["color"])
        part_names.append(name)

    doc = gltf.GLTF2()
    doc.asset = gltf.Asset(generator="Module5 make_temple_glb.py (procedural, deterministic)")
    scene = gltf.Scene(nodes=[])
    doc.scenes = [scene]
    doc.scene = 0

    binary_blob = b""
    for i, name in enumerate(part_names):
        pos = positions_list[i]
        norm = normals_list[i]
        idx = indices_list[i].astype(np.uint16)

        pos_bytes = pos.astype("<f4").tobytes()
        norm_bytes = norm.astype("<f4").tobytes()
        idx_bytes = idx.astype("<u2").tobytes()
        if len(idx_bytes) % 4 != 0:
            idx_bytes += b"\x00" * (4 - len(idx_bytes) % 4)  # pad to 4-byte boundary

        pos_offset = len(binary_blob)
        binary_blob += pos_bytes
        norm_offset = len(binary_blob)
        binary_blob += norm_bytes
        idx_offset = len(binary_blob)
        binary_blob += idx_bytes

        doc.bufferViews.append(gltf.BufferView(buffer=0, byteOffset=pos_offset, byteLength=len(pos_bytes), target=gltf.ARRAY_BUFFER))
        pos_bv = len(doc.bufferViews) - 1
        doc.bufferViews.append(gltf.BufferView(buffer=0, byteOffset=norm_offset, byteLength=len(norm_bytes), target=gltf.ARRAY_BUFFER))
        norm_bv = len(doc.bufferViews) - 1
        doc.bufferViews.append(gltf.BufferView(buffer=0, byteOffset=idx_offset, byteLength=len(idx.tobytes()), target=gltf.ELEMENT_ARRAY_BUFFER))
        idx_bv = len(doc.bufferViews) - 1

        doc.accessors.append(gltf.Accessor(
            bufferView=pos_bv, componentType=gltf.FLOAT, count=len(pos),
            type=gltf.VEC3, min=pos.min(axis=0).tolist(), max=pos.max(axis=0).tolist()))
        pos_acc = len(doc.accessors) - 1
        doc.accessors.append(gltf.Accessor(
            bufferView=norm_bv, componentType=gltf.FLOAT, count=len(norm), type=gltf.VEC3))
        norm_acc = len(doc.accessors) - 1
        doc.accessors.append(gltf.Accessor(
            bufferView=idx_bv, componentType=gltf.UNSIGNED_SHORT, count=len(idx), type=gltf.SCALAR))
        idx_acc = len(doc.accessors) - 1

        doc.materials.append(gltf.Material(
            name=name,
            pbrMetallicRoughness=gltf.PbrMetallicRoughness(
                baseColorFactor=list(colors[i]), metallicFactor=0.05, roughnessFactor=0.85)))
        mat_idx = len(doc.materials) - 1

        mesh = gltf.Mesh(name=name, primitives=[gltf.Primitive(
            attributes=gltf.Attributes(POSITION=pos_acc, NORMAL=norm_acc),
            indices=idx_acc, material=mat_idx)])
        doc.meshes.append(mesh)
        mesh_idx = len(doc.meshes) - 1

        doc.nodes.append(gltf.Node(mesh=mesh_idx, name=name))
        node_idx = len(doc.nodes) - 1
        scene.nodes.append(node_idx)

    doc.buffers.append(gltf.Buffer(byteLength=len(binary_blob)))
    doc.set_binary_blob(binary_blob)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    doc.save_binary(path)
    return doc


def write_hotspots(path):
    payload = {
        "model": "pusaka_temple.glb",
        "_comment": "Generic Hindu-Buddhist candi cosmology (kaki/tubuh/atap "
                    "= Bhurloka/Bhuvarloka/Svarloka; finial = Mount Meru) -- "
                    "not a claim about any specific named real-world site.",
        "hotspots": HOTSPOTS,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    glb_path = os.path.join(OUT_DIR, "pusaka_temple.glb")
    hotspots_path = os.path.join(OUT_DIR, "hotspots.json")
    write_glb(glb_path)
    write_hotspots(hotspots_path)
    size_kb = os.path.getsize(glb_path) / 1024.0
    print(f"wrote {glb_path} ({size_kb:.1f} KB)")
    print(f"wrote {hotspots_path} ({len(HOTSPOTS)} hotspots)")
