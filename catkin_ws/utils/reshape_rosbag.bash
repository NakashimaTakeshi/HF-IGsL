#!/usr/bin/env bash
BAG_DIRECTORY=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
BAGS=$(find ${BAG_DIRECTORY} -maxdepth 1 -name '*.bag' | awk '{print $0 " " substr($0,match($0,/[0-9]+/))}' | sort -k2 -n | cut -f 1 -d " ")

for bag in ${BAGS};
do
    echo "Processing bag file ${bag}"
    # https://gist.github.com/sven-bock/408b6e845666e06a0cf4002271c2780f
    rosbag filter ${bag} $(basename "${bag}" .bag)_filtered.bag "topic != '/tf' or (len(m.transforms)>0 and m.transforms[0].child_frame_id=='base_footprint')"
done

#  rosbag filter dataset1_2023-01-24-13-49-23.bag output.bag "topic != '/tf' or (len(m.transforms)>0 and m.transforms[0].child_frame_id=='base_footprint')"
