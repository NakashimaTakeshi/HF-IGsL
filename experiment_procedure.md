# Procedure for the experiment

This document is to describe how to execute HF-PGM experiment.

## Prerequisite
Complete the environment setup following the [README.md](README.md).

## Training
1.   Download rosbag data for training:
     ```shell
     wget --load-cookies ./tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ./tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1QWerhv0VN-_PGYYa3jimLW1_FDk2qIqw' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1QWerhv0VN-_PGYYa3jimLW1_FDk2qIqw" -O MCL+MRSSM_model2.zip\
     && rm -rf ./tmp/cookies.txt \
     && unzip -o ./MCL+MRSSM_model2.zip -d ./catkin_ws/src/ros_rssm/Multimodal-RSSM/train/HF-PGM/House/MRSSM/MRSSM/results \
     && rm -rf ./MCL+MRSSM_model2.zip

     ```
1.   Transform:
     ```shell
     wget --load-cookies ./tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ./tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1QWerhv0VN-_PGYYa3jimLW1_FDk2qIqw' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1QWerhv0VN-_PGYYa3jimLW1_FDk2qIqw" -O MCL+MRSSM_model2.zip\
     && rm -rf ./tmp/cookies.txt \
     && unzip -o ./MCL+MRSSM_model2.zip -d ./catkin_ws/src/ros_rssm/Multimodal-RSSM/train/HF-PGM/House/MRSSM/MRSSM/results \
     && rm -rf ./MCL+MRSSM_model2.zip

     ```
1.   Transform:
     ```shell
     wget --load-cookies ./tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ./tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1QWerhv0VN-_PGYYa3jimLW1_FDk2qIqw' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1QWerhv0VN-_PGYYa3jimLW1_FDk2qIqw" -O MCL+MRSSM_model2.zip\
     && rm -rf ./tmp/cookies.txt \
     && unzip -o ./MCL+MRSSM_model2.zip -d ./catkin_ws/src/ros_rssm/Multimodal-RSSM/train/HF-PGM/House/MRSSM/MRSSM/results \
     && rm -rf ./MCL+MRSSM_model2.zip

     ```


## Evaluation
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

1.   You can also conveniently run multiple terminals simultaneously using Terminator by running:

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


