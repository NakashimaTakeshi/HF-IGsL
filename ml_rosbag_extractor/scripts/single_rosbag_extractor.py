#!/usr/bin/env python
import os
import sys
import numpy as np
import rosbag
import warnings

from record_utils import *


if __name__ == '__main__':
    target_hz = 1
    target_topics = dict(
        image_hsr_256=dict(
            topic_name = "/camera/rgb/image_raw/compressed",
            topic_msg_type = "sensor_msgs/CompressedImage",
            image_options = dict(
                clip_mode = 1,
                clip_width = (80, 560),
                clip_height = (None,None),
                # clip_mode = 2,
                # clip_width = 320,
                # clip_height_center = 320,
                # clip_width_center = 370,
                resize = (258, 258),
            ),
            buf_clear = True,
        ),
        Pose=dict(
            topic_name = "/amcl_pose",
            topic_msg_type = "geometry_msgs/PoseWithCovarianceStamped",
            buf_clear = True,
        ),
    )

    bag_path = sys.argv[1]
    in_bag = rosbag.Bag(bag_path)

    t_s = in_bag.get_start_time()
    t_e = in_bag.get_end_time()

    print("t_s=",t_s)
    print("t_e=",t_e)

    read_msg_list = []
    for key in target_topics.keys():
        read_msg_list.append(target_topics[key]["topic_name"])

    recoder = TopicRecoder(target_hz, target_topics)

    if "sound" in target_topics.keys():
        recoder.init_audio(in_bag)

    # collect topic
    for topic, msg, t in in_bag.read_messages(topics=read_msg_list):
        percentage = (t.to_sec() - t_s) / (t_e - t_s)
        sys.stdout.write('\r{0:4.2f}%'.format(100 * percentage))
        t = t.to_sec() - t_s
        recoder(topic, msg, t)
    sys.stdout.flush()
    recoder.obs_complement()

    out_path = os.path.splitext(bag_path)[0]+"_1.npy"
    recoder.save_dataset(out_path)
    

    print('\nDone!')
