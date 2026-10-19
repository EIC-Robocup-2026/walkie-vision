"""Check 9: OSNet x1.0 person re-identification (walkie_vision.reid) on the GPU.

Three people cropped from bus.jpg: each crop must be closer to a shifted copy
of itself than to the other two people.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from common import assets_dir, expect, finish, gpu_mem_of, timed_ms
from walkie_vision.models import HUB_MODELS
from walkie_vision.reid import INPUT_SIZE, MEAN, STD, load_osnet

hub = HUB_MODELS["osnet"]
print(f"{hub.repo} at {hub.revision} ({hub.licence})")
model = load_osnet("cuda")
print("loaded; parameters", f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

im = Image.open(assets_dir() / "bus.jpg").convert("RGB")
people = [(50, 400, 250, 900), (220, 400, 350, 860), (670, 380, 810, 880)]
# The same three people, with each box moved by a few pixels.
shifted = [(x1 + 6, y1 + 10, x2 + 6, y2 - 4) for x1, y1, x2, y2 in people]
h, w = INPUT_SIZE
crops = [im.crop(b).resize((w, h), Image.BILINEAR) for b in people + shifted]
mean = torch.tensor(MEAN, device="cuda").view(1, 3, 1, 1)
std = torch.tensor(STD, device="cuda").view(1, 3, 1, 1)
x = torch.from_numpy(np.stack([np.asarray(c) for c in crops])).permute(0, 3, 1, 2).float().div(255).cuda()
x = (x - mean) / std
with torch.inference_mode():
    _, first = timed_ms(model, x)
    v, dt = timed_ms(model, x)
v = torch.nn.functional.normalize(v, dim=-1)
print(f"input {tuple(x.shape)} -> embeddings {tuple(v.shape)}; first call {first:.0f} ms, then {dt:.1f} ms for 6 crops")
sims = (v[:3] @ v[3:].T).cpu()
print("cosine, people x shifted people:", [[round(a, 3) for a in r] for r in sims.tolist()])
expect(v.shape == (6, 512), "512-value embeddings")
expect(bool((sims.argmax(dim=1) == torch.arange(3)).all()), "each person is closest to their own shifted crop")
print("GPU memory (nvidia-smi):", gpu_mem_of())

finish()
