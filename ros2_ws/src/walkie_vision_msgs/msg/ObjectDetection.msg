# One object that the open-vocabulary detector found in one camera image.
# The ObjectDetectionArray that holds it has the image's header and
# camera_info. Boxes are in pixels of the image that camera_info describes.

# The class name the detector matched: one of ObjectDetectionArray.classes.
string label
# The detector's confidence, from 0 to 1.
float32 score
# The tracker's id for this object. It stays the same from image to image
# while the track lasts. 0 means that the object is not tracked.
uint32 track_id
# The box around the object. center.theta is always 0.
vision_msgs/BoundingBox2D bbox

# The embedding of the image inside bbox. embedding_model names the weights
# that made it: it is the <model> part of walkie-database's vector table
# name, for example clip_vit_b16 for object_clip_vit_b16. The vector length
# is the length of embedding, and the vector has unit length. Both fields
# are empty when no embedding was computed.
string embedding_model
float32[] embedding

# The object's points, cut out of the depth image with the object's mask,
# in the frame given in points.header. Empty when there was no depth image.
sensor_msgs/PointCloud2 points
# The image inside bbox, compressed. Optional: data is empty when no crop
# was sent.
sensor_msgs/CompressedImage crop
