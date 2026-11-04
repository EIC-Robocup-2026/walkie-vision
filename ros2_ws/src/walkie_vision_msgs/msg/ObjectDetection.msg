# One observation. Pixel coordinates use the containing array's camera_info.
# No map pose, persistent object ID, segmentation mask or person record here.
string label                       # One member of ObjectDetectionArray.classes.
float32 score                      # Finite detector confidence in [0,1].
uint32 track_id                    # Hint within stream_id; 0 means untracked.
vision_msgs/BoundingBox2D bbox      # Original box in full-image pixels; theta=0.
# Image and text vectors share the exact weights/preprocessing named here.
# Both empty when not sampled; otherwise finite unit vector (CLIP: 512).
# Crop uses 15% padding PER SIDE, clipped to image, before model preprocessing.
string embedding_model
float32[] embedding
# XYZ float32 little-endian metres, x right/y down/z forward, height=1.
# Header equals array.header. Cap 2000 points. Empty data/width=0 if <30
# valid points or depth unavailable. Never re-stamp or use a map frame.
sensor_msgs/PointCloud2 points
# JPEG quality 90; same header. The padded crop, not just bbox pixels.
# Empty data when not sampled. Longest side <=512; never upscale.
sensor_msgs/CompressedImage crop
# Finite [0,1]; zero when disabled or unusable for best-view selection.
float32 view_quality
# Optional mask-only colour evidence, computed BEFORE JPEG/padding/resizing.
# Empty when not sampled or <50 valid mask pixels; otherwise 12 shares
# summing to 1: black,white,grey,red,orange,yellow,green,blue,purple,pink,brown,beige.
# This freezes the colour seam while masks still exist; no mask is transmitted.
float32[] color_shares
