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
#Command -maxdepth 1 makes sure it only searches in the folder and not subfolders
for bag in $(find ${BAG_DIRECTORY} -maxdepth 1 -name '*.bag'); 
do
	echo "Processing bag file ${bag}"
	INPUT_DATASET_FILE="${bag}"

    # record_all2.sh
    for((i=0; i<20; i++)); 
    do
        roslaunch ros_rssm rssm_amcl.launch file_1:=${bag} &
        sleep 6.0
        ROSLAUNCH_PID=${!}
        echo "roslaunch PID = ${ROSLAUNCH_PID}"
        movie_name=$(basename "${bag}" .bag)_${i}.mp4
        # https://ffmpeg.org/ffmpeg-all.html#x11grab
        ffmpeg -r film -y -an -f x11grab -show_region 1 -draw_mouse 0 -video_size 1850x1020 -framerate film -i :1.0+70,60 "${movie_name}" &
        
        wait $ROSLAUNCH_PID
        kill -TERM ${!}

    done

	echo "==============================================================================="
	sleep 1
done


