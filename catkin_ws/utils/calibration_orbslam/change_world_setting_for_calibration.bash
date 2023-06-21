# To install calibration pattern to aws for ORB-SLAM

#!/usr/bin/env bash
CURRENT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
cd ${CURRENT_DIR}
MODEL_PATH=/root/TurtleBot3/catkin_ws/src/aws_robomaker_small_house_world/models/
WORLD_PATH=/root/TurtleBot3/catkin_ws/src/aws_robomaker_small_house_world/worlds/

cp -r -f --backup=numbered ./calibration_pattern ${MODEL_PATH}
cp -f --backup=numbered small_house.world ${WORLD_PATH}