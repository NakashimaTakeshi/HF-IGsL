#!/bin/bash

################################################################################

# Download the YOLO pre-trained weights missing in the 'darknet_ros' package into the 'rgiro_launch' package.
# http://pjreddie.com/media/files/yolov2.weights
# http://pjreddie.com/media/files/yolo9000.weights
# http://pjreddie.com/media/files/yolov3.weights
mkdir -p /root/TurtleBot3/catkin_ws/src/rgiro_launch/config/darknet_ros/weights/
wget http://pjreddie.com/media/files/yolov2.weights -N -P /root/TurtleBot3/catkin_ws/src/rgiro_launch/config/darknet_ros/weights
wget http://pjreddie.com/media/files/yolo9000.weights -N -P /root/TurtleBot3/catkin_ws/src/rgiro_launch/config/darknet_ros/weights
wget http://pjreddie.com/media/files/yolov3.weights -N -P /root/TurtleBot3/catkin_ws/src/rgiro_launch/config/darknet_ros/weights

# Link the downloaded file paths.
ln -s /root/TurtleBot3/catkin_ws/src/rgiro_launch/config/darknet_ros/weights/* /root/TurtleBot3/catkin_ws/src/darknet_ros/darknet_ros/yolo_network_config/weights/

################################################################################

# Download the missing datasets in the 'rgiro_spco2_slam' package.

mkdir -p /root/TurtleBot3/catkin_ws/src/rgiro_spco2_slam/data/rosbag/albert-b-laser-vision/albert-B-laser-vision-dataset/
mkdir -p /root/TurtleBot3/catkin_ws/src/rgiro_spco2_slam/data/output/test/img/
mkdir -p /root/TurtleBot3/catkin_ws/src/rgiro_spco2_slam/data/output/test/map/
mkdir -p /root/TurtleBot3/catkin_ws/src/rgiro_spco2_slam/data/output/test/particle/
mkdir -p /root/TurtleBot3/catkin_ws/src/rgiro_spco2_slam/data/output/test/tmp/
mkdir -p /root/TurtleBot3/catkin_ws/src/rgiro_spco2_slam/data/output/test/weight/

wget https://raw.githubusercontent.com/a-taniguchi/SpCoSLAM2/master/rosbag/albert-b-laser-vision/albert-B-laser-vision-dataset/teaching.csv -N -P /root/TurtleBot3/catkin_ws/src/rgiro_spco2_slam/data/rosbag/albert-b-laser-vision/albert-B-laser-vision-dataset
wget https://github.com/a-taniguchi/SpCoSLAM2/raw/master/rosbag/albert-b-laser-vision/albert-B-laser-vision-dataset/CNN_Place365.zip -N -P /root/TurtleBot3/catkin_ws/src/rgiro_spco2_slam/data/rosbag/albert-b-laser-vision/albert-B-laser-vision-dataset

unzip -o /root/TurtleBot3/catkin_ws/src/rgiro_spco2_slam/data/rosbag/albert-b-laser-vision/albert-B-laser-vision-dataset/CNN_Place365.zip -d /root/TurtleBot3/catkin_ws/src/rgiro_spco2_slam/data/rosbag/albert-b-laser-vision/albert-B-laser-vision-dataset/
