# One complete successful detector result, including a successful empty frame.
# No publication for an inference failure. classes is the searched vocabulary.
std_msgs/Header header             # Colour image capture time and optical frame.
sensor_msgs/CameraInfo camera_info # Rectified calibration for bbox/points pixels.
string[] classes                   # Ordered unique nonempty class names.
ObjectDetection[] detections
# UUID generated once at publisher-process startup, including fake publisher.
# Identity key for a track is (source topic, stream_id, kind, track_id).
# Not a frame handle and cannot retrieve an old image. Empty is invalid.
string stream_id
