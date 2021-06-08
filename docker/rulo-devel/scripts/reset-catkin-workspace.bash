#!/bin/bash

################################################################################

# Download package lists from Ubuntu repositories.
apt-get update

# Install system dependencies required by specific ROS packages.
# http://wiki.ros.org/rosdep
rosdep update

# Source the updated ROS environment.
source /opt/ros/melodic/setup.bash

################################################################################

# Remove the Catkin workspace.

# Delete unexpected 'catkin_make' artefacts.
cd /root/RULO/catkin_ws/ && rm -r .catkin_workspace build/ devel/ install/
cd /root/RULO/catkin_ws/src/ && rm CMakeLists.txt

# Delete expected 'catkin build' artefacts.
cd /root/RULO/catkin_ws/ && catkin clean -y
cd /root/RULO/catkin_ws/ && rm -r CMakeLists.txt .catkin_tools/

################################################################################

# Initialize and build the Catkin workspace.
cd /root/RULO/catkin_ws/ && catkin_init_workspace && catkin build

# Source the Catkin workspace.
source /root/RULO/catkin_ws/devel/setup.bash
