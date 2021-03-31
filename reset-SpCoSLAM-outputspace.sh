#!/bin/bash


# Delete expected 'catkin_make' artefacts.
cd /root/RULO/catkin_ws/src/rgiro_spco2/rgiro_spco2_learning/data/output/test/ && rm -r *
mkdir /root/RULO/catkin_ws/src/rgiro_spco2/rgiro_spco2_learning/data/output/test/img
mkdir /root/RULO/catkin_ws/src/rgiro_spco2/rgiro_spco2_learning/data/output/test/map
mkdir /root/RULO/catkin_ws/src/rgiro_spco2/rgiro_spco2_learning/data/output/test/weight
mkdir /root/RULO/catkin_ws/src/rgiro_spco2/rgiro_spco2_learning/data/output/test/particle
mkdir /root/RULO/catkin_ws/src/rgiro_spco2/rgiro_spco2_learning/data/output/test/tmp

cd /root/RULO/catkin_ws/src/rgiro_spco2/rgiro_spco2_data/output/test/ && rm -r *
mkdir /root/RULO/catkin_ws/src/rgiro_spco2/rgiro_spco2_data/output/test/img
mkdir /root/RULO/catkin_ws/src/rgiro_spco2/rgiro_spco2_data/output/test/map
mkdir /root/RULO/catkin_ws/src/rgiro_spco2/rgiro_spco2_data/output/test/weight
mkdir /root/RULO/catkin_ws/src/rgiro_spco2/rgiro_spco2_data/output/test/particle
mkdir /root/RULO/catkin_ws/src/rgiro_spco2/rgiro_spco2_data/output/test/tmp
