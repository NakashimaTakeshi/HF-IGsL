import numpy as np

#npy for training
# TARGET="/root/TurtleBot3/catkin_ws/src/ros_rssm/Multimodal-RSSM/dataset/HF-PGM/MobileRobot_with_Image_Pose/Turtlebot3Image_20230125/train/2023-01-16-13-52-01_1.npy"

#npy for validation
# TARGET="/root/TurtleBot3/catkin_ws/src/ros_rssm/Multimodal-RSSM/dataset/HF-PGM/MobileRobot_with_Image_Pose/Turtlebot3Image_20230125/validation/2023-01-16-13-08-48_1.npy"
# TARGET="/root/TurtleBot3/catkin_ws/src/ros_rssm/Multimodal-RSSM/dataset/HF-PGM/MobileRobot_with_Image_Pose/Turtlebot3Image_20230125/validation/2023-01-16-14-03-58_1.npy"

#npy for states estimation
# TARGET="/root/TurtleBot3/catkin_ws/src/ros_rssm/Multimodal-RSSM/dataset/HF-PGM/MobileRobot_with_Image_Pose/Turtlebot3Image_20230125/validation/2023-01-16-13-08-48_1.npy"
TARGET="/root/TurtleBot3/catkin_ws/src/ros_rssm/Multimodal-RSSM/train/HF-PGM/House/MRSSM/MRSSM/results/HF-PGM_model2-seed_0/2023-03-30/run_0/states_models_3000.npy"

arr = np.load(TARGET, allow_pickle=True)
print(arr.shape)
print(arr.dtype)