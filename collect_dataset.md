# Procedure for the collecting data

This document is to describe how to collect data for experiment.
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
  Add gazebo plugin to publish robot pose ground truth for evaluation.(topic name:/tracker)  
  Change camera position of turtlebot3 1m higher.  

     ```shell
       bash /root/TurtleBot3/catkin_ws/utils/change_submodule_setting/change_submodule_setting.bash 
     ```
## Create map.
  Launch gazebo simulator and necessary nodes.

   ```shell
      roslaunch ros_rssm rssm_data_collect_slam.launch
      roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
   ```
  Save map after exploreing. 

   ```shell
      rosrun map_server map_saver -f [map_file_name]
   ```

  If necessary, modify the map.(there some useful [tools](https://github.com/naka-lab/ros_navigation) to edit map.)


## Recode rosbag.
### For traning and validate (M)RSSM.
   Launch gazebo simulator and necessary nodes.
   
   ```shell
   cd /root/TurtleBot3/dataset/recording_ws
   roslaunch ros_rssm rssm_data_collect.launch
   ```

  When the robot is operated with teleop.

   ```shell
   roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
   ```

  When the robot is operated with following waypoint.
  (we use ros [follow_waypoints](http://wiki.ros.org/follow_waypoints) Ros package)
  If not installed , install package. "apt-get install ros-kinetic-follow-waypoints"  
  Please follow the instructions displayed in the terminal.

   ```shell
   roslaunch follow_waypoints follow_waypoints.launch
   ```
  Sellect Pose for waypoints by '2D Pose Estimate' and Save waypoints list to csv by following command
     ```shell
      rostopic pub /path_ready std_msgs/Empty -1
     ```

  Record ROS bag data for trainig (M)RSSM
  Pose estimated amcl and obserbation(image) are necessary.
     ```shell
         rosbag record -a -x "(.*)/compressedDepth(.*)"
         rosbag record /amcl_pose /camera/color/image_raw/compressed /clock /tracker
     ```

 Convert rosbag to npy format for traning.

   ```shell
         cd /root/TurtleBot3/ml_rosbag_extractor/scripts/
         python3 single_rosbag_extractor.py [PathForROSbag] 
   ```

###  Record ROS bag data for HF-PGM evaluation  
  At least following topic is reqwired to store robot motion.
   ```shell
      rosbag record /tf /tf_static /tracker /scan /camera/color/image_raw/compressed 
   ```

   If the tf contains topics published by amcl at the time of data collection, they can later be removed with the rosbag filter command.
      # remove unwanted topics form rosbag file.
   ```shell
      cp /root/TurtleBot3/catkin_ws/utils/reshape_rosbag.bash /root/TurtleBot3/dataset/recording_ws
      bash reshape_rosbag.bash 
      rm /root/TurtleBot3/dataset/recording_ws/reshape_rosbag.bash
   ```

   Excute robot kidnap by calling following [rosservice call](https://classic.gazebosim.org/tutorials?tut=ros_comm&cat=connect_ros)
   
     ```shell
       rosservice call /gazebo/set_model_state '{model_state: { model_name: turtlebot3_waffle_pi, pose: { position: { x: 0.3, y: 0.2 ,z: 0 }, orientation: {x: 0, y: 0, z: 0.7071067812, w: 0.7071067812 } } } }'

      #  If you want to get current pose, use following command.
      rosservice call /gazebo/get_model_state "model_name: 'turtlebot3_waffle_pi' " 

     ```

