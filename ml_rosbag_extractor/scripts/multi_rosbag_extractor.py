#!/usr/bin/env python
import os
import sys
import numpy as np
import glob
import rosbag
from multiprocessing import Pool

from record_utils import *

bag_dir = "rosbags"
max_process = 2
target_hz = 10
target_topics = dict(
    image_horizon=dict(
        topic_name = "/camera_side/color/image_raw/compressed",
        topic_msg_type = "sensor_msgs/CompressedImage",
        image_options = dict(
            # clip_mode = 1,
            # clip_width = (None,None),
            # clip_height = (80, 560),
            clip_mode = 2,
            clip_width = 320,
            clip_height_center = 320,
            clip_width_center = 420,
            resize = (80, 80),
        ),
        buf_clear = True,
    ),
    image_vertical=dict(
        topic_name = "/camera_top/color/image_raw/compressed",
        topic_msg_type = "sensor_msgs/CompressedImage",
        image_options = dict(
            # clip_mode = 1,
            # clip_width = (280, 530),
            # clip_height = (250, 500),
            clip_mode = 2,
            clip_width = 200,
            clip_height_center = 340,
            clip_width_center = 450,
            resize = (80, 80),
        ),
        buf_clear = True,
    ),
    image_vertical_high_resolution=dict(
        topic_name = "/usb_cam/image_raw/compressed",
        topic_msg_type = "sensor_msgs/CompressedImage",
        image_options = dict(
            clip_mode = 2,
            clip_width = 320,
            clip_height_center = 915,
            clip_width_center = 1100,
            resize = (None, None),
        ),
        buf_clear = True,
    ),
    sound=dict(
        topic_name = "/audio/audio",
        topic_msg_type = "audio_common_msgs/AudioData",
        sound_info_topic = "/audio/audio_info",
        sound_option = dict(
            convert_mlsp = False,
            library = "torchaudio",
            device = "cuda:0",
        ),
        buf_clear = False,
    ),
    joint_states=dict(
        topic_name = "/arm2/joints/get/joint_states",
        topic_msg_type = "sensor_msgs/JointState",
        buf_clear = True,
        delta_option = True,
        delta_dict_key = "d_joint_states",
    ),
    pose_quat=dict(
        topic_name = "/arm2_kinematics/get/pose",
        topic_msg_type = "geometry_msgs/PoseStamped",
        buf_clear = True,
        rpy_option = True,
        rpy_dict_key = "pose_rpy",
    ),
)


def single_rosbag_process(bag_path):
    in_bag = rosbag.Bag(bag_path)
    t_s = in_bag.get_start_time()
    t_e = in_bag.get_end_time()
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

    out_path = os.path.splitext(bag_path)[0]+".npy"
    recoder.save_dataset(out_path)
    print('\nDone!')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        bag_dir = sys.argv[1]
    # search rosbags
    bag_list = glob.glob(os.path.join(bag_dir, '*.npy'))
    print("find %d npy files!" % len(bag_list))
    
    p = Pool(max_process)
    for bag_path in bag_list:
        p.apply_async(single_rosbag_process, args=(bag_path,))
    p.close()
    p.join()
    print('All subprocesses done.')
