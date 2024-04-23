#!/bin/bash

# sudo apt-get update
# sudo apt-get -y install ffmpeg

# DATA_DIR=/root/TurtleBot3/catkin_ws/result/eval/env2/movie/tmp
DATA_DIR=/root/TurtleBot3/catkin_ws/result/eval/env1_Reproduction/movie/tmp
UTILE_DIR=/root/TurtleBot3/catkin_ws/utils/analysis_tools

cd $UTILE_DIR
bash combine_mp4.bash $DATA_DIR dataset1 16 1 7 9
bash combine_mp4.bash $DATA_DIR dataset2 4  1 14 10
bash combine_mp4.bash $DATA_DIR dataset4 1  1 3 8
bash combine_mp4.bash $DATA_DIR dataset5 4  1 10 8
bash combine_mp4.bash $DATA_DIR dataset6 3  1 9 15
bash combine_mp4.bash $DATA_DIR dataset7 2  2 9 10
bash combine_mp4.bash $DATA_DIR dataset8 14 1 8 19
bash combine_mp4.bash $DATA_DIR dataset9 8  1 4 14
bash combine_mp4.bash $DATA_DIR dataset10 17 2 7 0
bash combine_mp4.bash $DATA_DIR dataset11 12 2 6 6
bash combine_mp4.bash $DATA_DIR dataset12 15 1 10 2

# bash combine_mp4.bash $DATA_DIR dataset1 0  12 11 18
# bash combine_mp4.bash $DATA_DIR dataset4 10 5  13 19
# bash combine_mp4.bash $DATA_DIR dataset5 17 11 18 13
# bash combine_mp4.bash $DATA_DIR dataset2 19 19 14 18
# bash combine_mp4.bash $DATA_DIR dataset3 1  8   5  1
# bash combine_mp4.bash $DATA_DIR dataset7 7  4   1 16
# bash combine_mp4.bash $DATA_DIR dataset8 14 4  12  8