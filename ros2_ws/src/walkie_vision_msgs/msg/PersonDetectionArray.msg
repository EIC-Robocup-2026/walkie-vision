# Every person found in one camera image.
#
# One array is published for every image that is processed, also when no
# person was found.

# The stamp and the camera frame of the image.
std_msgs/Header header
# The calibration of the image that the boxes and keypoints are in.
sensor_msgs/CameraInfo camera_info
PersonDetection[] people
