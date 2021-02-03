cmake_minimum_required(VERSION 2.8.3)
project(rulo_chatter)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS
  roscpp
  std_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(rulo_talker src/rulo_talker.cpp)
add_executable(rulo_listener src/rulo_listener.cpp)

target_link_libraries(rulo_talker ${catkin_LIBRARIES})
target_link_libraries(rulo_listener ${catkin_LIBRARIES})
