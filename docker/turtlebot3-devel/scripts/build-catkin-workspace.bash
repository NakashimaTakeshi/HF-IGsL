#!/bin/bash

################################################################################

# Source the ROS distribution environment.
source /opt/ros/melodic/setup.bash

# Build the Catkin workspace.
cd /root/TurtleBot3/catkin_ws/ && catkin build

# Source the Catkin workspace.
source /root/TurtleBot3/catkin_ws/devel/setup.bash
