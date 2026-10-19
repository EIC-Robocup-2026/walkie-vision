"""Check 6: Hugging Face models on the GPU, at the revisions pinned in walkie_vision.models.

Usage: python c06_transformers.py [siglip2|florence2|embeddinggemma2]

The models download into the Hugging Face cache on first use.
"""

from __future__ import annotations

import sys

import torch
from PIL import Image

from common import assets_dir, expect, finish, gpu_mem_of, timed_ms
from walkie_vision.models import HUB_MODELS

which = sys.argv[1] if len(sys.argv) > 1 else "siglip2"
hub = HUB_MODELS[which]
print(f"{hub.repo} at {hub.revision} ({hub.licence})")
bus = Image.open(assets_dir() / "bus.jpg").convert("RGB")
zidane = Image.open(assets_dir() / "zidane.jpg").convert("RGB")


def as_tensor(out):
    """transformers 5 returns a ModelOutput from get_*_features."""
    return out if isinstance(out, torch.Tensor) else out.pooler_output


if which == "siglip2":
    import transformers
    from transformers import AutoModel, AutoProcessor

    print("transformers", transformers.__version__)
    captions = [
        "a photo of a bus on a city street",
        "a photo of two men in suits",
        "a photo of a cat on a sofa",
        "a photo of a bowl of fruit on a kitchen table",
        "a photo of a sponge next to a sink",
    ]
    proc = AutoProcessor.from_pretrained(hub.repo, revision=hub.revision)
    for dtype in (torch.float16, torch.float32):
        model = AutoModel.from_pretrained(hub.repo, revision=hub.revision, dtype=dtype).to("cuda").eval()
        with torch.inference_mode():
            img_in = proc(images=[bus, zidane], return_tensors="pt").to("cuda")
            img_in["pixel_values"] = img_in["pixel_values"].to(dtype)
            _, first = timed_ms(model.get_image_features, **img_in)
            out, t_img = timed_ms(model.get_image_features, **img_in)
            img = as_tensor(out)
            # SigLIP 2 was trained with text padded to 64 tokens.
            txt_in = proc(text=captions, padding="max_length", max_length=64, return_tensors="pt").to("cuda")
            txt = as_tensor(model.get_text_features(**txt_in))
        img = torch.nn.functional.normalize(img.float(), dim=-1)
        txt = torch.nn.functional.normalize(txt.float(), dim=-1)
        sims = img @ txt.T
        print(f"SigLIP 2 {dtype}: embedding size {img.shape[-1]}; 2 images: first call {first:.0f} ms, then {t_img:.1f} ms")
        for name, row, want in (("bus.jpg", sims[0], 0), ("zidane.jpg", sims[1], 1)):
            best = int(row.argmax())
            print(f"  {name}: '{captions[best]}'; similarities {[round(v, 3) for v in row.tolist()]}")
            expect(best == want, f"{dtype}: {name} matches '{captions[want]}'")
        print("  GPU memory (nvidia-smi):", gpu_mem_of())
        del model
        torch.cuda.empty_cache()

elif which == "florence2":
    import transformers
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print("transformers", transformers.__version__)
    model = AutoModelForImageTextToText.from_pretrained(hub.repo, revision=hub.revision, dtype=torch.float16)
    model = model.to("cuda").eval()
    proc = AutoProcessor.from_pretrained(hub.repo, revision=hub.revision)
    print("model class", type(model).__name__, "from", type(model).__module__)
    task = "<CAPTION>"
    captions = {}
    for name, im in (("bus.jpg", bus), ("zidane.jpg", zidane)):
        inputs = proc(text=task, images=im, return_tensors="pt").to("cuda")
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
        with torch.inference_mode():
            ids, dt = timed_ms(model.generate, **inputs, max_new_tokens=40, num_beams=3, do_sample=False)
        text = proc.batch_decode(ids, skip_special_tokens=False)[0]
        captions[name] = proc.post_process_generation(text, task=task, image_size=im.size)[task]
        print(f"  {name}: {captions[name]!r} [{dt:.0f} ms]")
    expect("bus" in captions["bus.jpg"].lower(), "the bus.jpg caption mentions the bus")
    expect(bool(captions["zidane.jpg"].strip()), "the zidane.jpg caption is not empty")
    print("  GPU memory (nvidia-smi):", gpu_mem_of())

elif which == "embeddinggemma2":
    import sentence_transformers
    from sentence_transformers import SentenceTransformer

    print("sentence-transformers", sentence_transformers.__version__)
    # bf16 only where the GPU has it in hardware; Turing emulates it slowly.
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float32
    model = SentenceTransformer(
        hub.repo,
        revision=hub.revision,
        device="cuda",
        model_kwargs={"dtype": dtype},
        # The audio tower is not needed for text and images.
        config_kwargs={"audio_config": None},
    )
    print(f"loaded {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters as {next(model.parameters()).dtype}")
    texts = ["a bus on a city street", "two men in suits", "a cat on a sofa"]
    temb, t_txt = timed_ms(model.encode, texts, prompt_name="Retrieval-query", convert_to_tensor=True)
    iemb, t_img = timed_ms(model.encode, [{"image": bus}, {"image": zidane}], convert_to_tensor=True)
    print(f"text embeddings {tuple(temb.shape)} [{t_txt:.0f} ms], image embeddings {tuple(iemb.shape)} [{t_img:.0f} ms]")
    expect(bool(torch.isfinite(temb).all() and torch.isfinite(iemb).all()), "embeddings are finite")
    sims = model.similarity(iemb, temb)
    print("image x text cosine:", [[round(v, 3) for v in row] for row in sims.tolist()])
    expect(int(sims[0].argmax()) == 0, "bus.jpg matches 'a bus on a city street'")
    expect(int(sims[1].argmax()) == 1, "zidane.jpg matches 'two men in suits'")
    print("GPU memory (nvidia-smi):", gpu_mem_of())

finish()
