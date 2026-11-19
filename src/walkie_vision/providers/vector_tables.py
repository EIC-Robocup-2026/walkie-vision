"""The vector model names that walkie-database stores, and the vector length of each.

It imports nothing from walkie_vision, so every provider and every message
check can import it without a cycle.
"""

# The model names of walkie-database's vector tables (its ADR-0007).
# room_clip_vit_b16 and object_clip_vit_b16 name only the network; there is
# one set of CLIP ViT-B/16 weights in use, OpenAI's. The face and appearance
# names follow walkie-database's migration 0002; before it, its tables are
# face_arcface and appearance_osnet.
CLIP_VIT_B16 = "clip_vit_b16"
ARCFACE_W600K_R50 = "arcface_w600k_r50"
OSNET_X1_0_MSMT17 = "osnet_x1_0_msmt17"
FAKE_MODEL_PREFIX = "fake_"
"""A fake's model name starts with this, so nothing reads a fake vector as real."""
EMBEDDING_DIMS = {CLIP_VIT_B16: 512}
"""The vector length of each real object embedder. Other names, such as a fake's, are not checked."""
