cmake_minimum_required(VERSION 3.8)
project(turtlebot3_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)   # the .srv code generator

# the three services this lean repo needs
set(srv_files
  "srv/DrlStep.srv"
  "srv/Goal.srv"
  "srv/RingGoal.srv"
)

# turn the .srv text files into Python/C++ types.
# (No DEPENDENCIES line: all fields are primitives, so no other interface pkgs are referenced.)
rosidl_generate_interfaces(${PROJECT_NAME}
  ${srv_files}
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()   # MUST be last
