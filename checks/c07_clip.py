"""Check 7: CLIP ViT-B/16 (OpenAI weights) with open_clip on the GPU.

The OpenAI weights were trained with QuickGELU, so they are loaded into the
"ViT-B-16-quickgelu" config; the plain "ViT-B-16" config uses GELU and gives
slightly wrong embeddings.
"""

from __future__ import annotations

import open_clip
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from common import assets_dir, expect, finish, gpu_mem_of, timed_ms
from walkie_vision.models import HUB_MODELS

hub = HUB_MODELS["clip"]
print("open_clip", open_clip.__version__, f"| {hub.repo} at {hub.revision} ({hub.licence})")
weights = hf_hub_download(hub.repo, "open_clip_model.safetensors", revision=hub.revision)
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16-quickgelu", pretrained=weights, device="cuda")
model.eval()
tok = open_clip.get_tokenizer("ViT-B-16-quickgelu")
texts = ["a photo of a bus", "a photo of two men", "a photo of a cat"]
ims = torch.stack([preprocess(Image.open(assets_dir() / n).convert("RGB")) for n in ("bus.jpg", "zidane.jpg")]).cuda()
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
    _, first = timed_ms(model.encode_image, ims)
    fi, dt = timed_ms(model.encode_image, ims)
    ft = model.encode_text(tok(texts).cuda())
fi = torch.nn.functional.normalize(fi.float(), dim=-1)
ft = torch.nn.functional.normalize(ft.float(), dim=-1)
sims = fi @ ft.T
print(f"image embeddings {tuple(fi.shape)}, text embeddings {tuple(ft.shape)}; 2 images: first call {first:.0f} ms, then {dt:.1f} ms")
print("image x text cosine:", [[round(v, 3) for v in row] for row in sims.tolist()])
expect(int(sims[0].argmax()) == 0, "bus.jpg matches 'a photo of a bus'")
expect(int(sims[1].argmax()) == 1, "zidane.jpg matches 'a photo of two men'")
print("GPU memory (nvidia-smi):", gpu_mem_of())

finish()
