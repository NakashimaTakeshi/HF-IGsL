#!/usr/bin/env bash
CURRENT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
cd ${CURRENT_DIR}

AMCL_PATH=/root/TurtleBot3/catkin_ws/src/turtlebot3/turtlebot3_navigation/launch/
# XACRO_PATH1=
XACRO_PATH=/root/TurtleBot3/catkin_ws/src/turtlebot3/turtlebot3_description/urdf/

cp -f --backup=numbered amcl.launch turtlebot3_navigation.launch ${AMCL_PATH}
cp -f --backup=numbered turtlebot3_waffle_pi.urdf.xacro turtlebot3_waffle_pi.gazebo.xacro ${XACRO_PATH}