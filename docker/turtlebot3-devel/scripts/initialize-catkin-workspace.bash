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

# Initialize and build the Catkin workspace.
cd /root/RULO/catkin_ws/ && catkin_init_workspace && catkin build

# Source the Catkin workspace.
source /root/RULO/catkin_ws/devel/setup.bash
