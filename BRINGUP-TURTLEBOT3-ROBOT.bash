#!/bin/bash

# This script connects by SSH to the TurtleBot3 robot(s) and executes the commands to:
# 1. Kill all existing ROS processes inside the robot(s).
# 2. Execute the launch files that bring up the robot(s).
#
# Usage: bash BRINGUP-TURTLEBOT3-ROBOT.bash

################################################################################

ssh -f -t -l pi turtlebot3-01.local 'bash -c "
source /opt/ros/kinetic/setup.bash;
source ~/catkin_ws/devel/setup.bash;
export ROS_MASTER_URI=http://sr-station-04.local:11311;
export ROS_HOSTNAME=turtlebot3-01.local;
export TURTLEBOT3_MODEL=waffle_pi;
killall -9 roscore;
killall -9 rosmaster;
#/usr/sbin/ntpdate ntp.ubuntu.com;
ROS_NAMESPACE=turtlebot3_01 roslaunch turtlebot3_bringup turtlebot3_robot.launch multi_robot_name:="turtlebot3_01" set_lidar_frame_id:="turtlebot3_01/base_scan" &
sleep 3;
ROS_NAMESPACE=turtlebot3_01 roslaunch turtlebot3_bringup turtlebot3_rpicamera.launch &
sleep 3;
rosrun image_transport republish compressed in:=/turtlebot3_01/raspicam_node/image image_transport:=compressed raw out:=/turtlebot3_01/raspicam_node/image &
sleep 3;
echo "Done"
"'

ssh -f -t -l pi turtlebot3-02.local 'bash -c "
source /opt/ros/kinetic/setup.bash;
source ~/catkin_ws/devel/setup.bash;
export ROS_MASTER_URI=http://sr-station-04.local:11311;
export ROS_HOSTNAME=turtlebot3-02.local;
export TURTLEBOT3_MODEL=waffle_pi;
killall -9 roscore;
killall -9 rosmaster;
#/usr/sbin/ntpdate ntp.ubuntu.com;
ROS_NAMESPACE=turtlebot3_02 roslaunch turtlebot3_bringup turtlebot3_robot.launch multi_robot_name:="turtlebot3_02" set_lidar_frame_id:="turtlebot3_02/base_scan" &
sleep 3;
ROS_NAMESPACE=turtlebot3_02 roslaunch turtlebot3_bringup turtlebot3_rpicamera.launch &
sleep 3;
echo "Done"
"'

ssh -f -t -l pi turtlebot3-03.local 'bash -c "
source /opt/ros/kinetic/setup.bash;
source ~/catkin_ws/devel/setup.bash;
export ROS_MASTER_URI=http://sr-station-04.local:11311;
export ROS_HOSTNAME=turtlebot3-03.local;
export TURTLEBOT3_MODEL=waffle_pi;
killall -9 roscore;
killall -9 rosmaster;
#/usr/sbin/ntpdate ntp.ubuntu.com;
ROS_NAMESPACE=turtlebot3_03 roslaunch turtlebot3_bringup turtlebot3_robot.launch multi_robot_name:="turtlebot3_03" set_lidar_frame_id:="turtlebot3_03/base_scan" &
sleep 3;
ROS_NAMESPACE=turtlebot3_03 roslaunch turtlebot3_bringup turtlebot3_rpicamera.launch &
sleep 3;
echo "Done"
"'
