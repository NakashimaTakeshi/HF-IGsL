# Procedure for the experiment

This document is to describe how to execute HF-PGM experiment.

## Prerequisite
Complete the environment setup following the [README.md](README.md).

## Training
1.   Download rosbag data for training:
     ```shell
     TBD

     ```
1.   Transform:
     ```shell
TBD

     ```
1.   download npy files for training:
     ```shell
     wget --load-cookies ./tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ./tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1AnmvoF3wyUZ0rdBvHn4YXzN9McHeMzlj' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1AnmvoF3wyUZ0rdBvHn4YXzN9McHeMzlj" -O training_and_validation_data.zip\
     && rm -rf ./tmp/cookies.txt \
     && unzip -o ./training_and_validation_data.zip -d ./catkin_ws/src/ros_rssm/Multimodal-RSSM/dataset/HF-PGM/MobileRobot_with_Image_Pose/Turtlebot3Image_20230125 \
     && rm -rf ./training_and_validation_data.zip
     ```
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
1.   If you use pre-trained models, please download parameters of a neural network (weights) and config files([hydra](https://hydra.cc/docs/intro/) format):
     ```shell
     cd ./TurtleBot3
     
     wget --load-cookies ./tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ./tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1QWerhv0VN-_PGYYa3jimLW1_FDk2qIqw' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1QWerhv0VN-_PGYYa3jimLW1_FDk2qIqw" -O MCL+MRSSM_model2.zip\
     && rm -rf ./tmp/cookies.txt \
     && unzip -o ./MCL+MRSSM_model2.zip -d ./catkin_ws/src/ros_rssm/Multimodal-RSSM/train/HF-PGM/House/MRSSM/MRSSM/results \
     && rm -rf ./MCL+MRSSM_model2.zip

     wget --load-cookies ./tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ./tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=10aRCQYKsat7AOP1QvJ5Q3SvG4Gxq6cSI' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=10aRCQYKsat7AOP1QvJ5Q3SvG4Gxq6cSI" -O MCL+RSSM_model1.zip\
     && rm -rf ./tmp/cookies.txt \
     && unzip  -o ./MCL+RSSM_model1.zip -d ./catkin_ws/src/ros_rssm/Multimodal-RSSM/train/HF-PGM/House/MRSSM/MRSSM/results \
     && rm -rf ./MCL+RSSM_model1.zip

     wget --load-cookies ./tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ./tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1VWelpjpJC6HV-cD8DZcGpWqj57i9SWDM' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1VWelpjpJC6HV-cD8DZcGpWqj57i9SWDM" -O RSSM_baseline.zip\
     && rm -rf ./tmp/cookies.txt \
     && unzip  -o ./RSSM_baseline.zip -d ./catkin_ws/src/ros_rssm/Multimodal-RSSM/train/HF-PGM/House/MRSSM/MRSSM/results \
     && rm -rf ./RSSM_baseline.zip
     ```
<!--
        ```shell
        wget --no-check-certificate  'https://drive.google.com/uc?export=download&id=1QWerhv0VN-_PGYYa3jimLW1_FDk2qIqw' -O ./MCL+MRSSM_model2.zip
        ```
     The above command will not work because the file size is too large to run a google drive virus scan. 
     So bypassing virus check, referring to the following site
        https://gist.github.com/iamtekeste/3cdfd0366ebfd2c0d805?permalink_comment_id=3557609#gistcomment-3557609
        https://chadrick-kwag.net/wget-google-drive-large-files-bypassing-virus-check/
-->

  
2.   Download the dataset for evaluation and unzip it in the `./Turtlebot3/dataset` directory:
     ```shell
     wget --load-cookies ./tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ./tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1KDpE1EuH27lUXtBURONeB9SeIaleiUXr' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1KDpE1EuH27lUXtBURONeB9SeIaleiUXr" -O eval_dataset.zip\
     && rm -rf ./tmp/cookies.txt \
     && unzip  -o ./eval_dataset.zip -d . \
     && rm -rf ./eval_dataset.zip
     ```


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
     bash start_experiment.bash dataset
     ```
     You can change integration mode [here](https://gitlab.com/emlab/TurtleBot3/-/blob/HF-PGM_MRSSM-otake/catkin_ws/src/ros_rssm/scripts/RSSM_node_MRSSM.py#L272). (1:[use RSSM likelihopod](https://gitlab.com/emlab/TurtleBot3/-/blob/HF-PGM_nakashima/catkin_ws/src/navigation/amcl/src/amcl/sensors/amcl_laser.cpp#L337) /2:[replase 25% particle](https://gitlab.com/emlab/TurtleBot3/-/blob/HF-PGM_nakashima/catkin_ws/src/navigation/amcl/src/amcl/sensors/amcl_laser.cpp#L247) based on RSSM estimation )
     You can change model 1 or model 2 [here](https://gitlab.com/emlab/TurtleBot3/-/blob/HF-PGM_MRSSM-otake/catkin_ws/src/ros_rssm/launch/rssm_amcl.launch#L12). model detail is described in the paper.(coming soon)


     Then npy files that include beleief h_t and states s_t are saved in the directory specified in the [RSSM_node.py](https://gitlab.com/emlab/TurtleBot3/-/blob/HF-PGM_MRSSM-otake/catkin_ws/src/ros_rssm/scripts/RSSM_node.py#L93) or [RSSM_node_MRSSM.py](https://gitlab.com/emlab/TurtleBot3/-/blob/HF-PGM_MRSSM-otake/catkin_ws/src/ros_rssm/scripts/RSSM_node_MRSSM.py#L104).

## Evaluation
1.   copy bash and python files for evaluation from [this directory](https://gitlab.com/emlab/TurtleBot3/-/tree/HF-PGM_MRSSM-otake/ex_data/JSAI/log_model2_particle_dataset) to directory  that log npy files are
1.   execute bash twice in above directory.

     ```shell
       cd <YOURE DIRECTORY>
       bash prepare_eval.bash
       bash make_graph.bash 
       python3 eval_localization_csv.py
     ```
