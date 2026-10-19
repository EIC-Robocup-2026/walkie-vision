# One person found in one camera image, with their body pose, face and
# appearance. The PersonDetectionArray that holds it has the image's header
# and camera_info. Boxes and keypoints are in pixels of the image that
# camera_info describes.

# The person detector's confidence, from 0 to 1.
float32 score
# The tracker's id for this person. It stays the same from image to image
# while the track lasts. 0 means that the person is not tracked.
uint32 track_id
# The box around the whole person. center.theta is always 0.
vision_msgs/BoundingBox2D bbox

# The 17 COCO body keypoints, in COCO order: nose, left eye, right eye,
# left ear, right ear, left shoulder, right shoulder, left elbow, right
# elbow, left wrist, right wrist, left hip, right hip, left knee, right
# knee, left ankle, right ankle. keypoints holds x0, y0, x1, y1, and so on.
float32[34] keypoints
# The confidence of each keypoint, from 0 to 1. The position of a keypoint
# with a low score is a guess, for example when it is hidden.
float32[17] keypoint_scores

# The person's face, when one was found inside bbox. When there is no face,
# face_det_score is 0, and face_model and face_embedding are empty.
vision_msgs/BoundingBox2D face_bbox
float32 face_det_score
# face_model names the weights that made face_embedding: it is the <model>
# part of walkie-database's vector table name, for example
# arcface_w600k_r50 for face_arcface_w600k_r50. The vector length is the
# length of face_embedding, and the vector has unit length.
string face_model
float32[] face_embedding

# The appearance (re-identification) embedding of the image inside bbox,
# for recognising the person when the face is not visible. The model name,
# for example osnet_x1_0_msmt17, and the vector follow the same rules as
# the face fields. Both are empty when no embedding was computed.
string appearance_model
float32[] appearance_embedding
