#!/usr/bin/env bash
################################################################
# Usage, bash extract_all_rosbags_data.sh DIRECTORY, this will extract the data from all the rosbags in DIRECTORY
# This will only extract in the files inside the folder and will not look for subfolders, in case you want to extract rosbags inside subfolders, please modify the maxdepth parameter or erase it
####################################################################
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

# Temporarily setting the internal field seperator (IFS) to the newline character.
IFS=$'\n';

# Recursively loop through all bag files in the specified directory
BAG_DIRECTORY=$1
# BAGS=$(find ${BAG_DIRECTORY} -maxdepth 1 -name '*.bag' | awk '{print $0 " " substr($0,match($0,/[0-9]+/),RLENGTH)}' | sort -k2 -n | cut -f 1 -d " ")
BAGS=$(find ${BAG_DIRECTORY} -maxdepth 1 -name '*.bag' | awk '{print $0 " " substr($0,match($0,/[0-9]+/))}' | sort -k2 -n | cut -f 1 -d " ")

SAVE_DIR="/root/TurtleBot3/catkin_ws/result/eval/movie/"
if [ ! -d ${SAVE_DIR} ]; then
  mkdir ${SAVE_DIR}
fi
$(find ${SAVE_DIR} -maxdepth 1 -name "*.mp4" -type f -delete)

RSSM_MODEL=$2
# RSSM_MODEL="${RSSM_MODEL:- }"

echo "RSSM_MODEL = ${RSSM_MODEL}"

#Command -maxdepth 1 makes sure it only searches in the folder and not subfolders
for bag in ${BAGS}; 
do
	echo "Processing bag file ${bag}"
	INPUT_DATASET_FILE="${bag}"

    # record_all2.sh
    for((i=0; i<2; i++)); 
    do
        # rosclean purge -y
        roslaunch ros_rssm rssm_amcl.launch file_1:=${bag} rssm_model:=${RSSM_MODEL} &
        echo "roslaunch ros_rssm rssm_amcl.launch file_1:=${bag}"
        # roslaunch ros_rssm rssm_amcl.launch file_1:=${bag} &
        sleep 6.0
        ROSLAUNCH_PID=${!}
        echo "roslaunch PID = ${ROSLAUNCH_PID}"
        movie_name=$(basename "${bag}" .bag)_$(basename "${RSSM_MODEL}" .py)_${i}.mp4
        # https://ffmpeg.org/ffmpeg-all.html#x11grab
        # ffmpeg -r film -y -an -f x11grab -show_region 1 -draw_mouse 0 -video_size 1850x1020 -framerate film -i :1.0+70,60 "${SAVE_DIR}""${movie_name}" &
        ffmpeg -r film -y -an -f x11grab -show_region 1 -draw_mouse 0 -loglevel 8 -video_size 1850x1020 -framerate film -i :1.0+70,60 "${SAVE_DIR}""${movie_name}" &
        
        wait $ROSLAUNCH_PID
        kill -TERM ${!}

    done

	echo "==============================================================================="
	sleep 1
done


