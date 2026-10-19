# ADR-0006: Run the face models on onnxruntime without the insightface package

<!-- --8<-- [start:body] -->
**Status:** Accepted · **Date:** 2026-10-19

**Context**

The face models are InsightFace's `buffalo_l` pack: SCRFD for detection and
ArcFace (`w600k_r50`) for recognition, as ONNX files. The `insightface`
package on PyPI runs them, but:

- it requires `onnxruntime`, the CPU-only package. It installs the same
  `onnxruntime` Python module as `onnxruntime-gpu`, and can replace the CUDA
  provider without an error. Removing the requirement with a Pixi
  dependency override makes Pixi 0.80 treat `pixi.lock` as out of date, so
  `pixi install --locked` fails;
- it pulls in scikit-image, used only for one similarity transform;
- the PyPI 2.1 wheel has no licence file.

Only three of its files are needed: the SCRFD detector, the ArcFace
recogniser and the face alignment.

**Decision**

- `src/walkie_vision/face/` vendors `scrfd.py`, `arcface.py` and
  `face_align.py` from InsightFace commit `668e8d4` (v0.7.3). The code is
  MIT. Each file keeps the licence, the source URL and a list of changes.
- The changes: the caller creates the ONNX Runtime sessions, with explicit
  providers; a numpy version of the Umeyama similarity transform replaces
  scikit-image's.
- `FaceAnalyzer` in `analysis.py` is new code. It calls
  `onnxruntime.preload_dlls()` when CUDA is requested, and raises if a
  session does not run on the provider it asked for.
- `pixi run fetch-models` downloads `buffalo_l.zip` from InsightFace's
  release and checks it, and each file in it, against a SHA-256.
- `check-env` (`c02_packages`) fails if `onnxruntime`, `insightface` or
  scikit-image is installed.

**Consequences**

- The results match the package: against InsightFace 2.1 on the CPU, boxes
  within 1.5e-5 px and embeddings with a cosine of 0.9999998 or more.
- We maintain about 530 lines of vendored code. An upstream fix does not
  arrive by itself: compare with InsightFace's repository before a season
  starts.
- The `buffalo_l` models are for non-commercial research only. Vendoring the
  code does not change that.
- The other models in `buffalo_l` (landmarks, gender and age) are
  downloaded but not loaded.

**See also:** [Step 4: The face models](/setup/01-vision-environment.md#step-4-the-face-models-without-the-insightface-package),
[InsightFace](https://github.com/deepinsight/insightface)
<!-- --8<-- [end:body] -->
