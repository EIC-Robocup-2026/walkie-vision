cmake_minimum_required(VERSION 3.10)
project(walkie_vision_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ObjectDetection.msg"
  "msg/ObjectDetectionArray.msg"
  "msg/PersonDetection.msg"
  "msg/PersonDetectionArray.msg"
  "srv/EmbedText.srv"
  "srv/GetPersonCrop.srv"
  DEPENDENCIES std_msgs sensor_msgs vision_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
