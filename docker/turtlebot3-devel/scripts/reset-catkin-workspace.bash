#!/bin/bash

################################################################################

# Download package lists from Ubuntu repositories.
apt-get update

# Install system dependencies required by specific ROS packages.
# http://wiki.ros.org/rosdep
rosdep update

################################################################################

# Remove the Catkin workspace.

# Delete unexpected 'catkin_make' artefacts.
cd /root/TurtleBot3/catkin_ws/ && rm -r .catkin_workspace build/ devel/ install/
cd /root/TurtleBot3/catkin_ws/src/ && rm CMakeLists.txt

# Delete expected 'catkin build' artefacts.
cd /root/TurtleBot3/catkin_ws/ && catkin clean -y
cd /root/TurtleBot3/catkin_ws/ && rm -r CMakeLists.txt .catkin_tools/

################################################################################

# Initialize the Catkin workspace.
cd /root/TurtleBot3/catkin_ws/ && catkin_init_workspace

# Build the Catkin workspace.
source /root/TurtleBot3/docker/turtlebot3-devel/scripts/build-catkin-workspace.bash
