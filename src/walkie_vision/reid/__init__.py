"""Person re-identification: OSNet x1.0, vendored from deep-person-reid (MIT).

load_osnet() returns the network with Kaiyang Zhou's MSMT17 re-ID weights.
It maps a person crop to a 512-value appearance embedding; compare two
embeddings by cosine similarity. walkie_vision.reid.identifier wraps it as a
walkie_vision.providers.ReIdentifier (OSNetReIdentifier). It is not imported
here, because it imports this module.
"""

from __future__ import annotations

import torch

from ..models import HUB_MODELS, OSNET_CHECKPOINT
from .osnet import osnet_x1_0

# The input the checkpoint was trained with: RGB crops resized to 256x128
# (height x width), scaled to [0, 1] and normalised with these values.
INPUT_SIZE = (256, 128)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def load_osnet(device: str | torch.device = "cuda") -> torch.nn.Module:
    """Return OSNet x1.0 with the MSMT17 re-ID weights, in eval mode.

    In eval mode the network returns embeddings, not identity scores. The
    weights download into the Hugging Face cache on first use.
    """
    from huggingface_hub import hf_hub_download

    hub = HUB_MODELS["osnet"]
    path = hf_hub_download(hub.repo, OSNET_CHECKPOINT, revision=hub.revision)
    # weights_only: the file must hold plain tensors, so no pickled code runs.
    state = torch.load(path, map_location="cpu", weights_only=True)
    # The classifier scores MSMT17's 4101 training identities; embeddings
    # come from the layer before it, so it is left out.
    state = {k: v for k, v in state.items() if not k.startswith("classifier.")}
    model = osnet_x1_0(num_classes=1)
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys or set(result.missing_keys) != {"classifier.weight", "classifier.bias"}:
        raise RuntimeError(f"{OSNET_CHECKPOINT} does not fit osnet_x1_0: {result}")
    return model.to(device).eval()
