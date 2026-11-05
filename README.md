# Module 5 -- AI Graphics: NeRF, World Models & AR (reference repo)

Course: *Grafika Komputer / Computer Graphics* (`25MP7716G030`), S1 Informatika
semester 5. **Integrates Modules 1-4 and is the capstone dry run.** The course
reference `Module5_EN.tex` is not included in this standalone repository.

**Rule:** the pinhole ray generator, volume-rendering compositing formula,
and full differentiable-rendering training loop are **NumPy only** (no
PyTorch/JAX/autograd). The backward pass is hand-derived and checked against
finite differences in `tests.py`. The **AR app** (`ar_app/`) is a static web
page using `<model-viewer>` and vanilla JS. Browser rendering and interaction
are verified; physical-device AR and occlusion are not yet verified.

## Quick start

Run from the repository root, `C:\Users\venerdi\Documents\pusaka3d-ar`,
not a `code/Module5` subdirectory. These are sequential PowerShell commands;
check that each command succeeds before continuing.

```powershell
python -m pip install -r requirements.txt
python ray_generator.py             # E1: pinhole ray through pixel (400,240)
python volume_render.py             # E2: composite 3 samples -> 1 pixel
python capture.py                   # writes data/pusaka_capture/ (8 synthetic views)
python fit_nerf_toy.py               # trains the original scene; train/novel PSNR
python nerf_bug.py                   # 2 clustered views vs 8 spread views
python tests.py                     # ALL CHECKS PASSED (17 ok)
python sigma_init_sweep.py           # original 8 views; post-update MSE + plot/JSON
python extended_scene.py             # separate temple-plus-sphere capture and fit
python tests.py                     # recheck original behavior after extension
```

The intentional negative control is a separate command. **Exit code 1 is
expected**, not a successful test run; it demonstrates detection of a broken
backward pass.

```powershell
python tests.py --break-on-purpose
$LASTEXITCODE                       # expected: 1
```

Generate and serve the AR app from the repository root:

```powershell
python ar_app/make_temple_glb.py
python -m http.server 8743 --directory ar_app
```

Open `http://localhost:8743/index.html`. Keep the server running. In a second
PowerShell terminal at the repository root, install and run browser checks:

```powershell
python -m pip install playwright
python -m playwright install chromium
python ar_app/verify_ar_app.py       # ALL CHECKS PASSED (10 ok)
```

The verifier launches Playwright with `channel="chromium"`, not the Chromium
headless shell. For a physical-device check, use a phone-accessible deployment
with the secure-context requirements of the chosen AR mode, then test the
"View in your space" handoff. A successful desktop browser check does not
establish physical AR support.

## Unit table

| File | Job | Reproduces |
|---|---|---|
| `ray_generator.py` | pinhole camera: pixel -> camera-space ray direction, and batched `generate_rays()` | Running example E1 |
| `volume_render.py` | density -> alpha, front-to-back compositing along a ray | Running example E2 |
| `nerf_toy.py` | analytic scene, look-at camera, learnable `VoxelGrid` with nearest-voxel lookup and softplus/sigmoid activations, hand-derived backward pass, training loop | From-scratch mini-NeRF |
| `capture.py` | deterministic 8-view ring capture (`poses.json` + PPM images) | Synthetic input |
| `fit_nerf_toy.py` | fits the original capture; reports train/novel-view PSNR and writes comparison PPMs | Reconstruction evidence |
| `nerf_bug.py` | 2 clustered views vs 8 spread views, same held-out novel view | View-coverage diagnose-and-fix |
| `tests.py` | E1-E2, finite-difference gradient check, view-coverage regression; optional intentional failure | Numerical checks |
| `sigma_init_sweep.py` | compares raw-density initializations with original fitting settings; writes MSE histories, final post-update MSE, plot and JSON | Initialization experiment |
| `extended_scene.py` | separate temple-plus-sphere capture and fit; checks original scene remains numerically unchanged | Own scene extension |
| `ar_app/make_temple_glb.py` | procedural temple GLB + 5 bilingual educational hotspots | AR model and content |
| `ar_app/index.html`, `app.js`, `style.css` | model viewer, AR handoff button, hotspots, quiz, EN/ID toggle | Pusaka3D AR Explainer |
| `ar_app/verify_ar_app.py` | Playwright Chromium browser checks and screenshots | Browser verification, not physical AR |

## What `python tests.py` asserts

- **E1:** pixel `(400,240)`, `f=100`, principal point `(320,240)` gives unnormalised camera-space direction `(0.8, 0, -1)`, normalised approximately `(0.625, 0, -0.781)`.
- **E2:** samples with alpha `(0.5, 0.5, 1.0)` and red/green/blue colours give weights `(0.5, 0.25, 0.25)` and pixel colour `(0.5, 0.25, 0.25)`; weights plus background sum to `1.0`.
- **Analytic vs finite-difference gradient:** hand-derived `backward_grid()` density and colour gradients match central differences within `1e-6`. This is a numerical gradient check, not comparison against an external autodiff library.
- **View coverage:** 8 spread views beat 2 views clustered `20 deg` apart by more than `1 dB` on the same held-out novel view at `190 deg`.

## Initialization and view coverage

The current initialization sweep uses the original 8-view fitting settings.
MSE is averaged over all views, pixels and RGB channels. History entries are
**pre-update**; the final values below are recomputed **after all updates**.

| Initial `sigma_raw` | Final post-update MSE (2026-11-05) |
|---|---|
| `-5` | `0.0479716661` |
| `-2` | `0.0082160265` |
| `0` | `0.0091180473` |
| `1` | `0.0152715282` |

At `-5`, the softplus derivative is `sigmoid(-5) = 0.0066928509`, which
attenuates density gradients. This initialization performs worse in the
current sweep, but **training does not stall entirely under these settings**.
The measured sweep does not support the old claim of completely stalled
training. See `out/sigma_init_sweep.json`, `out/sigma_init_sweep.png`, and
`out/evidence/sigma_init_sweep.txt`.

The view-coverage experiment evaluates both models at `190 deg`, far from
the clustered pair. An angle near that pair would instead favour its local
interpolation and would not test the same coverage limitation. The current
measured held-out improvement is `2.71 dB`; it is a toy-scene comparison,
not a guarantee for arbitrary scenes or camera layouts.

## Asset provenance

The starting implementation comes from the **provided course reference**;
this repository adds its own opt-in **sphere extension** to the analytic
scene. The capture images are procedurally rendered synthetic views, and
`ar_app/make_temple_glb.py` procedurally generates the temple GLB and hotspot
data. **No external downloaded mesh is used.** These provenance statements
do not establish a license for the supplied course material; its licensing
is not known here.

## Known deviations from the course plan

1. **Synthetic fallback, not real capture or a GPU NeRF pipeline.** `data/pusaka_capture/` is a deterministic analytic-scene capture, not phone photographs. Reconstruction uses a hand-written NumPy voxel grid and analytic gradients, not nerfstudio, COLMAP or Instant-NGP. The extension writes separately to `data/pusaka_capture_extended/`.
2. **Nearest-voxel lookup, not trilinear interpolation.** Each sample scatters its gradient into one voxel via `np.add.at`, rather than a weighted set of eight. This simplifies the hand-derived backward pass at the cost of blockier reconstruction. Toy-scale PSNR is not evidence of photorealism.
3. **Procedural model, not scanned heritage geometry.** The GLB contains six flat-coloured primitives. It is not a reconstruction of a real building; see the provenance section rather than assuming an asset license.
4. **No physical AR or occlusion verification yet.** Playwright checks browser rendering and interaction up to the AR handoff. Android Scene Viewer, iOS Quick Look, WebXR placement, real-world scale, anchoring and occlusion still require physical-device testing. Record device and browser evidence when performed.
5. **Generic educational content.** Hotspot text describes general Hindu-Buddhist candi concepts rather than documenting a specific captured heritage site.
6. **External CDN dependency.** The app loads `<model-viewer>` from an external CDN. Network/CDN availability is required; this is not a fully offline, self-contained deployment.
7. **Course reference absent in standalone checkout.** `Module5_EN.tex` and the parent course figure-generation pipeline are not included here. No TeX compilation or course-figure verification is claimed for the current run.

## Verified

Actual recorded run: **2026-11-05**. Toolchain: platform
`Windows-10-10.0.26200-SP0`, Python **3.11.9**, NumPy **2.3.5**,
matplotlib **3.10.9**, pygltflib **1.16.5**, Playwright **1.64.1**.
Browser verification uses **`channel="chromium"`**, not headless shell, with
SwiftShader/WebGL launch flags in `ar_app/verify_ar_app.py`.

| Item | Actual result (2026-11-05) | Evidence |
|---|---|---|
| Normal numerical tests | **17 checks passed**, including recheck after extension | `out/evidence/tests.txt`, `out/evidence/tests_after_extension.txt` |
| Analytic vs finite-difference gradient | Maximum error **`4.8854482959104217e-11`** | `out/evidence/tests.txt` |
| Intentional broken backward pass | Mismatch **`0.78161352450862231`**; **expected exit code 1** | `out/evidence/tests_break_on_purpose.txt` |
| Ray generation / volume compositing | Ran successfully | `out/evidence/ray_generator.txt`, `out/evidence/volume_render.txt` |
| Original capture fit | Mean training-view / novel-view PSNR **20.69 / 14.44 dB** | `out/evidence/fit_nerf_toy.txt` |
| View-coverage diagnose-and-fix | Buggy train/novel **21.03 / 9.30 dB**; fixed train/novel **19.49 / 12.02 dB**; held-out gap **2.71 dB** | `out/evidence/nerf_bug.txt` |
| Extended temple-plus-sphere scene | Mean training-view / novel-view PSNR **19.717741 / 14.915443 dB**; original scene numerically unchanged | `out/evidence/extended_scene.txt`, `out/extended_comparison.png` |
| Density initialization sweep | Final post-update MSE values listed above | `out/evidence/sigma_init_sweep.txt`, `out/sigma_init_sweep.json`, `out/sigma_init_sweep.png` |
| Procedural GLB generation | Ran successfully | `out/evidence/make_temple_glb.txt` |
| AR browser verification | **10 checks passed**: rendering, hotspots, information panel, language toggle, quiz and no JS errors | `out/evidence/verify_ar_app.txt` |
| Browser screenshots | Overview, English/Indonesian info panels, quiz mode | `ar_app/screenshots/m5_ar_overview.png`, `ar_app/screenshots/m5_ar_infopanel_en.png`, `ar_app/screenshots/m5_ar_infopanel_id.png`, `ar_app/screenshots/m5_ar_quizmode.png` |
| Physical-device AR / occlusion | **Pending; not verified** | No physical-device evidence yet |
| Clean-clone reproduction | **Passed** at `2dfe21425d90f427259a8c8a74bdc62809a40bab`: fresh local Git clone, isolated venv, requirements install, GLB generation, **17 numerical checks passed**; generated assets unchanged | `out/evidence/clean_clone.txt`; Python **3.11.9**, NumPy **2.4.6**, matplotlib **3.11.2**, pygltflib **1.16.5**; browser and physical-device AR not rerun |
