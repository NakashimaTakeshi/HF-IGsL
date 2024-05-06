# Procedure for the experiment

This document is to describe how to execute HF-PGM experiment.

## Prerequisite
Complete the environment setup following the [setup_env.md](setup_env.md).

## Training

1.   Download [npy](https://drive.google.com/file/d/1uVPE1vWM5bMVslnOpipVS-ldjWvckBTh/view?usp=drive_link) files for training and validation and place it `/root/TurtleBot3/catkin_ws/src/ros_rssm/Multimodal-RSSM/dataset/HF-PGM/MobileRobot_with_Image_Pose/`.
1.   Start training
     ```shell
     cd /root/TurtleBot3/catkin_ws/src/ros_rssm/Multimodal-RSSM/train/HF-PGM/House/MRSSM/MRSSM/
     python3 main.py
     ```
1.   Check training agent 
     ```shell
     cp <TRAINING_RESULT_PATH>/hydra_config.yaml /root/TurtleBot3/catkin_ws/src/ros_rssm/Multimodal-RSSM/train/HF-PGM/House/MRSSM/MRSSM/eval_targets/hoge/
     python3 estimate_state.py

     "before run following python script you have to change hard cording PATH"
     python3 check_model_ver2_MRSSM.py
     ```


## Execution
1.   If you use pre-trained models, please download [parameters](https://drive.google.com/file/d/1kQFJCbMcX-ewVZjSNmDQGpsHatH3YaWo/view?usp=drive_link) of a neural network (weights) and put it ./catkin_ws/src/ros_rssm/Multimodal-RSSM/train/HF-PGM/House/MRSSM/MRSSM/results:

  
2.   Download the [dataset](https://drive.google.com/file/d/1snB7aMvaPxUAKFGFw3PPDGV7ZmJ6X7Md/view?usp=drive_link) for evaluation and unzip it in the `./Turtlebot3/dataset/processed/` directory:

     ```shell
     cd ./TurtleBot3/ && bash ./RUN-TERMINATOR-TERMINAL.bash simulation
     or
     cd ./TurtleBot3/ && bash ./RUN-DOCKER-CONTAINER.bash
     ```
1.   Build package:

     ```shell
     sde-build-catkin-workspace
     ```

1.   execute:

     ```shell
     bash start_experiment.bash dataset/processed/env2 RSSM_node_MRSSM.py
     bash start_experiment.bash dataset/processed/env2 RSSM_node.py
     bash start_experiment.bash dataset/processed/env2 record_amcl.py
     ```
     or
     ```shell
     manage_experiments.bash
     ```

     You can change integration mode [here](https://gitlab.com/emlab/TurtleBot3/-/blob/HF-PGM_nakashima/catkin_ws/src/ros_rssm/scripts/RSSM_node_MRSSM.py?ref_type=heads#L276). (1:[use RSSM likelihopod](https://gitlab.com/emlab/TurtleBot3/-/blob/HF-PGM_nakashima/catkin_ws/src/navigation/amcl/src/amcl/sensors/amcl_laser.cpp#L337) /2:[replase 25% particle](https://gitlab.com/emlab/TurtleBot3/-/blob/HF-PGM_nakashima/catkin_ws/src/navigation/amcl/src/amcl/sensors/amcl_laser.cpp#L247) based on RSSM estimation )
     You can change model 1 or model 2 [here](https://gitlab.com/emlab/TurtleBot3/-/blob/HF-PGM_nakashima/catkin_ws/src/ros_rssm/launch/rssm_amcl.launch?ref_type=heads#L15). 

     Then npy files that include beleief h_t and states s_t are saved in the directory specified in the [RSSM_node.py](https://gitlab.com/emlab/TurtleBot3/-/blob/HF-PGM_nakashima/catkin_ws/src/ros_rssm/scripts/RSSM_node.py?ref_type=heads#L96) or [RSSM_node_MRSSM.py](https://gitlab.com/emlab/TurtleBot3/-/blob/HF-PGM_nakashima/catkin_ws/src/ros_rssm/scripts/RSSM_node_MRSSM.py?ref_type=heads#L106).

## Evaluation
1.   Evaluate the result.

     ```shell
       cd <YOURE DIRECTORY>
       bash /root/TurtleBot3/catkin_ws/utils/analysis_tools/prepare_eval.bash <YOURE DIRECTORY/npy>
       python3 /root/TurtleBot3/catkin_ws/utils/analysis_tools/eval_localization_csv.py  <YOURE DIRECTORY/npy>
     ```
