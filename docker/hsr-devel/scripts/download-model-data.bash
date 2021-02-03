#!/bin/bash

################################################################################

# Toyota HSR OwnCloud credentials.
# https://share.hsr.io/
HSR_OWNCLOUD_USER="???"
HSR_OWNCLOUD_PASSWORD="???"

################################################################################

# Download the YOLO pre-trained weights missing in the 'darknet_ros' package into the 'hsr_launch' package.
# http://pjreddie.com/media/files/yolov2.weights
# http://pjreddie.com/media/files/yolo9000.weights
# http://pjreddie.com/media/files/yolov3.weights
mkdir -p /root/HSR/catkin_ws/src/hsr_launch/config/darknet_ros/weights/
wget --http-user=${HSR_OWNCLOUD_USER} --http-password=${HSR_OWNCLOUD_PASSWORD} https://share.hsr.io/remote.php/dav/files/sdewg/data/HSR/darknet_ros/yolov2.weights -N -P /root/HSR/catkin_ws/src/hsr_launch/config/darknet_ros/weights
wget --http-user=${HSR_OWNCLOUD_USER} --http-password=${HSR_OWNCLOUD_PASSWORD} https://share.hsr.io/remote.php/dav/files/sdewg/data/HSR/darknet_ros/yolo9000.weights -N -P /root/HSR/catkin_ws/src/hsr_launch/config/darknet_ros/weights
wget --http-user=${HSR_OWNCLOUD_USER} --http-password=${HSR_OWNCLOUD_PASSWORD} https://share.hsr.io/remote.php/dav/files/sdewg/data/HSR/darknet_ros/yolov3.weights -N -P /root/HSR/catkin_ws/src/hsr_launch/config/darknet_ros/weights

# Link the downloaded file paths.
ln -s /root/HSR/catkin_ws/src/hsr_launch/config/darknet_ros/weights/* /root/HSR/catkin_ws/src/darknet_ros/darknet_ros/yolo_network_config/weights/
