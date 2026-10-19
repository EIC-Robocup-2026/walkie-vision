# Every object that the detector found in one camera image.
#
# One array is published for every image the detector processes, also when
# it found nothing. An array without a detection of a class in classes tells
# walkie-database that no such object was visible in this image.

# The stamp and the camera frame of the image.
std_msgs/Header header
# The calibration of the image that the boxes are in.
sensor_msgs/CameraInfo camera_info
# The class names that the detector looked for in this image: its text
# prompts. A class in this list without a detection was not seen.
string[] classes
ObjectDetection[] detections
