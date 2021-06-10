#!/bin/bash

################################################################################

# Download the YOLO pre-trained weights missing in the 'darknet_ros' package into the 'rulo_launch' package.
# http://pjreddie.com/media/files/yolov2.weights
# http://pjreddie.com/media/files/yolo9000.weights
# http://pjreddie.com/media/files/yolov3.weights
mkdir -p /root/RULO/catkin_ws/src/rulo_launch/config/darknet_ros/weights/
wget http://pjreddie.com/media/files/yolov2.weights -N -P /root/RULO/catkin_ws/src/rulo_launch/config/darknet_ros/weights
wget http://pjreddie.com/media/files/yolo9000.weights -N -P /root/RULO/catkin_ws/src/rulo_launch/config/darknet_ros/weights
wget http://pjreddie.com/media/files/yolov3.weights -N -P /root/RULO/catkin_ws/src/rulo_launch/config/darknet_ros/weights

# Link the downloaded file paths.
ln -s /root/RULO/catkin_ws/src/rulo_launch/config/darknet_ros/weights/* /root/RULO/catkin_ws/src/darknet_ros/darknet_ros/yolo_network_config/weights/

################################################################################

# Download the missing datasets in the 'rgiro_spco2_slam' package.

mkdir -p /root/RULO/catkin_ws/src/rgiro_spco2_slam/data/rosbag/albert-b-laser-vision/albert-B-laser-vision-dataset/
mkdir -p /root/RULO/catkin_ws/src/rgiro_spco2_slam/data/output/test/img/
mkdir -p /root/RULO/catkin_ws/src/rgiro_spco2_slam/data/output/test/map/
mkdir -p /root/RULO/catkin_ws/src/rgiro_spco2_slam/data/output/test/particle/
mkdir -p /root/RULO/catkin_ws/src/rgiro_spco2_slam/data/output/test/tmp/
mkdir -p /root/RULO/catkin_ws/src/rgiro_spco2_slam/data/output/test/weight/

wget https://raw.githubusercontent.com/a-taniguchi/SpCoSLAM2/master/rosbag/albert-b-laser-vision/albert-B-laser-vision-dataset/teaching.csv -N -P /root/RULO/catkin_ws/src/rgiro_spco2_slam/data/rosbag/albert-b-laser-vision/albert-B-laser-vision-dataset
wget https://github.com/a-taniguchi/SpCoSLAM2/raw/master/rosbag/albert-b-laser-vision/albert-B-laser-vision-dataset/CNN_Place365.zip -N -P /root/RULO/catkin_ws/src/rgiro_spco2_slam/data/rosbag/albert-b-laser-vision/albert-B-laser-vision-dataset

unzip -o /root/RULO/catkin_ws/src/rgiro_spco2_slam/data/rosbag/albert-b-laser-vision/albert-B-laser-vision-dataset/CNN_Place365.zip -d /root/RULO/catkin_ws/src/rgiro_spco2_slam/data/rosbag/albert-b-laser-vision/albert-B-laser-vision-dataset/
