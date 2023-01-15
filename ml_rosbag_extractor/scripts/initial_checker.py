#!/usr/bin/env python
# -*- coding: utf-8 -*-
# import sys
# sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
import os
import rospy
import numpy as np

from obs_utils import *
from topic_utils import *

from geometry_msgs.msg import TwistStamped, PoseStamped
from sensor_msgs.msg import Image, CompressedImage
from audio_common_msgs.msg import AudioData
from std_msgs.msg import Float32
from and_scale_ros.msg import WeightStamped

target_topics = dict(
    image_horizon=dict(
        topic_name = "/camera_side/color/image_raw/compressed",
        topic_msg_type = "sensor_msgs/CompressedImage",
        topic_msg_imp = CompressedImage,
        # image_option = dict(
        #     # clip_mode = 1,
        #     # clip_width = (None,None),
        #     # clip_height = (80, 560),
        #     clip_mode = 2,
        #     clip_width = 320,
        #     clip_height_center = 320,
        #     clip_width_center = 420,
        #     resize = (80, 80),
        # ),
        callback_option = dict(
            buffer_type = "variable",
            buffer_mode = "swap",
        ),
        extract_option = dict(
            buf_clear = True,
        ),
    ),
    image_vertical=dict(
        topic_name = "/camera_top/color/image_raw/compressed",
        topic_msg_type = "sensor_msgs/CompressedImage",
        topic_msg_imp = CompressedImage,
        # image_option = dict(
        #     # clip_mode = 1,
        #     # clip_width = (280, 530),
        #     # clip_height = (250, 500),
        #     clip_mode = 2,
        #     clip_width = 200,
        #     clip_height_center = 340,
        #     clip_width_center = 450,
        #     resize = (80, 80),
        # ),
        callback_option = dict(
            buffer_type = "variable",
            buffer_mode = "swap",
        ),
        extract_option = dict(
            buf_clear = True,
        ),
    ),
    # image_vertical_high_resolution=dict(
    #     topic_name = "/usb_cam/image_raw/compressed",
    #     topic_msg_type = "sensor_msgs/CompressedImage",
    #     topic_msg_imp = "sensor_msgs.msg.CompressedImage",
    #     image_option = dict(
    #         clip_mode = 2,
    #         clip_width = 320,
    #         clip_height_center = 915,
    #         clip_width_center = 1100,
    #         resize = (None, None),
    #     ),
    #     callback_option = dict(
    #         buffer_type = "variable",
    #         buffer_mode = "swap",
    #     ),
    #     extract_option = dict(
    #         buf_clear = True,
    #     ),
    # ),
    # sound=dict(
    #     topic_name = "/audio/audio",
    #     topic_msg_type = "audio_common_msgs/AudioData",
    #     topic_msg_imp = AudioData,
    #     sound_info_topic = "/audio/audio_info",
    #     topic_checker_option = dict(
    #         check_type = ["all_zero"],
    #     ),
    #     sound_option = dict(
    #         convert_mlsp = True,
    #         library = "torchaudio",
    #         device = "cuda:0",
    #         convert_pcm = True,
    #         pcm_bit_depth = 16,
    #     ),
    #     callback_option = dict(
    #         buffer_type = "list",
    #         buffer_length = 1600,
    #         buffer_mode = "extend",
    #     ),
    #     extract_option = dict(
    #         buf_clear = False,
    #     ),
    # ),
    # joint_states=dict(
    #     topic_name = "/arm2/joints/get/joint_states",
    #     topic_msg_type = "sensor_msgs/JointState",
    #     topic_msg_imp = "sensor_msgs.msg.JointState",
    #     callback_option = dict(
    #         buffer_type = "variable",
    #         buffer_mode = "swap",
    #     ),
    #     extract_option = dict(
    #         buf_clear = True,
    #         delta_option = True,
    #         delta_dict_key = "d_joint_states",
    #     ),
    # ),
    # pose_quat=dict(
    #     topic_name = "/arm2_kinematics/get/pose",
    #     topic_msg_type = "geometry_msgs/PoseStamped",
    #     topic_msg_imp = "geometry_msgs.msg.PoseStamped",
    #     callback_option = dict(
    #         buffer_type = "variable",
    #         buffer_mode = "swap",
    #     ),
    #     extract_option = dict(
    #         buf_clear = True,
    #         rpy_option = True,
    #         rpy_dict_key = "pose_rpy",
    #     ),
    # ),
)

def generate_diff_image(img1:np.ndarray, img2:np.ndarray):
    # check shape
    if img1.shape != img2.shape:
        raise ValueError("shapes of two images is not same")
    # check uint8
    if not (img1.dtype is np.uint8 and img2.dtype is np.uint8):
        raise ValueError("dtype of image must be np.uint8")
    # diff_img = img1.astype(np.int16) - img2.astype(np.int16)
    diff_img = img1 - img2
    return diff_img

def generate_alphablend_image(img1:np.ndarray, img2:np.ndarray):
    # check shape
    if img1.shape != img2.shape:
        raise ValueError("shapes of two images is not same")
    # check uint8
    if not (img1.dtype is np.uint8 and img2.dtype is np.uint8):
        raise ValueError("dtype of image must be np.uint8")
    blend_image = np.array(img1*0.5 + img2*0.5, dtype=np.uint8)
    return blend_image

class DiffImagePublisher():
    def __init__(self, target_topics, npy_path):
        self.target_topics = target_topics
        self.obs_maker = ObservationMaker(topic_cfg=target_topics)
        if os.path.exists(npy_path):
            self.base_data = np.load(npy_path, allow_pickle=True, encoding="latin1").item()
        else:
            # make dummy data
            self.base_data = dict()
            dummy_shape = (1080, 1920)
            for key in target_topics:
                self.base_data[key] = np.zeros(dummy_shape)
        
        self.pub_side_image = rospy.Publisher("/side_image_diff", Image, queue_size=1)
        self.pub_top_image = rospy.Publisher("/top_image_diff", Image, queue_size=1)

    def __call__(self):
        obs = self.obs_maker()
        for key in self.target_topics.keys():
            # diff_image = generate_diff_image(obs[key], self.base_data[key])
            diff_image = generate_alphablend_image(obs[key], self.base_data[key])
            if "side" in key:
                self.pub_side_image.publish(image2msg(diff_image))
            if "top" in key:
                self.pub_top_image.publish(image2msg(diff_image))


def main():
    npy_path = "cobotta_2022-05-29_point_drilling_rosbag_first.npy"
    rospy.init_node("InitialChekerNode")
    rate = rospy.Rate(10.0)
    generater = DiffImagePublisher(target_topics, npy_path)
    # rospy.loginfo("wait 10s")
    # rospy.sleep(5.0)
    rospy.loginfo("wait 5s")
    rospy.sleep(1.0)
    rospy.loginfo("4")
    rospy.sleep(1.0)
    rospy.loginfo("3")
    rospy.sleep(1.0)
    rospy.loginfo("2")
    rospy.sleep(1.0)
    rospy.loginfo("1")
    rospy.sleep(1.0)
    rospy.loginfo("start now!")
    while not rospy.is_shutdown():
        generater()
        rate.sleep()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass

