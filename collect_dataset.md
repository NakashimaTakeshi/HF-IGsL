# Procedure for the collecting data

This document is to describe how to execute HF-PGM experiment.
Dataset are collected in Gazebo.

## Prerequisite
Complete the environment setup following the [README.md](README.md).

## Setting
Change following from default setting 
1. amcl setting  
  Fix number of particles to 2000  
  Fix odom_model_type from "diff" to "diff-corrected"  
  <!-- Add "tf_broadcast" parameter to control publishing tf -->
1. turtlebot3 (xacro file) setting  
  Change odometrySource tab "world" to "encoder" to simulate kidnapped robot problem.  
  Add gazebo plugin to publish robot pose ground truth(topic name:/tracker)  
  Change camera position of turtlebot3 1m higher.  

     ```shell
       bash /root/TurtleBot3/catkin_ws/utils/change_submodule_setting/change_submodule_setting.bash 
     ```
## Recode rosbag.
     ```shell
       cd /root/TurtleBot3/dataset/recording_ws
       roslaunch ros_rssm rssm_data_collect.launch
      #  We'll remove tf frame(map to odom) that amcl node published.  
      #  - rosnode kill amcl
      #  - roslaunch turtlebot3_navigation amcl.launch tf_broadcast:=false
       rosbag record -a -x "(.*)/compressedDepth(.*)"

      #  Excute robot kidnap by calling following rosservice call 
      #  https://classic.gazebosim.org/tutorials?tut=ros_comm&cat=connect_ros
       rosservice call /gazebo/set_model_state '{model_state: { model_name: turtlebot3_waffle_pi, pose: { position: { x: 0., y: 0. ,z: 0. } } } }'
       rosservice call /gazebo/set_model_state '{model_state: { model_name: turtlebot3_waffle_pi, pose: { position: { x: 0.3, y: 0.2 ,z: 0 }, orientation: {x: 0, y: 0, z: 0.7071067812, w: 0.7071067812 } } } }'
      #  If you want to get current pose, use following command.
      #  rosservice call /gazebo/get_model_state "model_name: 'turtlebot3_waffle_pi' " 

      # remove unwanted topics form rosbag file.
       cp /root/TurtleBot3/catkin_ws/utils/reshape_rosbag.bash /root/TurtleBot3/dataset/recording_ws
       bash reshape_rosbag.bash 
       rm /root/TurtleBot3/dataset/recording_ws/reshape_rosbag.bash
     ```

