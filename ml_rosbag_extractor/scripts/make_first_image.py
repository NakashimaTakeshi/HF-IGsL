#!/usr/bin/env python
import os
import sys
import numpy as np
import rosbag

from record_utils import *

def convert_topic(msg, msg_type):
    if msg_type == "sensor_msgs/Image":
        data = image_raw_msg2opencv(msg)
    elif msg_type == "sensor_msgs/CompressedImage":
        data = image_cmp_msg2opencv(msg)
    elif msg_type == "audio_common_msgs/AudioData":
        data = float32_to_pcm(np.frombuffer(msg.data, dtype=np.float32))
    elif msg_type == "sensor_msgs/JointState":
        data = jointstate_converter(msg)["position"]
    elif msg_type == "geometry_msgs/PoseWithCovarianceStamped":
        data = posewithcovariancestamped_converter(msg)["pose"]
    elif msg_type == "geometry_msgs/PoseStamped":
        data = posestamped_converter(msg)["pose"]
    elif msg_type == "geometry_msgs/Vector3Stamped":
        data = vector3stamped_converter(msg)["vector"]
    elif msg_type == "geometry_msgs/TwistStamped":
        data = twiststamped_converter(msg)["twist"]
    elif msg_type == "and_scale_ros/WeightStamped":
        data = weight_stamped_converter(msg)["weight"][0]
    else:
        raise NotImplementedError("please check the type of topic")
    return data


def main():
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
                clip_width_center = 370,
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
        # image_vertical_high_resolution=dict(
        #     topic_name = "/usb_cam/image_raw/compressed",
        #     topic_msg_type = "sensor_msgs/CompressedImage",
        #     image_options = dict(
        #         clip_mode = 2,
        #         clip_width = 320,
        #         clip_height_center = 915,
        #         clip_width_center = 1100,
        #         resize = (None, None),
        #     ),
        #     buf_clear = True,
        # ),
        # sound=dict(
        #     topic_name = "/audio/audio",
        #     topic_msg_type = "audio_common_msgs/AudioData",
        #     sound_info_topic = "/audio/audio_info",
        #     sound_option = dict(
        #         convert_mlsp = False,
        #         library = "torchaudio",
        #         device = "cuda:0",
        #     ),
        #     buf_clear = False,
        # ),
        # joint_states=dict(
        #     topic_name = "/arm2/joints/get/joint_states",
        #     topic_msg_type = "sensor_msgs/JointState",
        #     buf_clear = True,
        #     # delta_option = True,
        #     # delta_dict_key = "d_joint_states",
        # ),
        # pose_quat=dict(
        #     topic_name = "/arm2_kinematics/get/pose",
        #     topic_msg_type = "geometry_msgs/PoseStamped",
        #     buf_clear = True,
        #     # rpy_option = True,
        #     # rpy_dict_key = "pose_rpy",
        # ),
        # desired_pose_quat=dict(
        #     topic_name = "/arm2_kinematics/set/desired_pose",
        #     topic_msg_type = "geometry_msgs/PoseStamped",
        #     buf_clear = True,
        #     # rpy_option = True,
        #     # rpy_dict_key = "pose_rpy",
        # ),
        # servo_value = dict(
        #     topic_name = "/servo_server/delta_twist_cmds",
        #     topic_msg_type = "geometry_msgs/TwistStamped",
        #     buf_clear = True
        # ),
        # weight_value = dict(
        #     topic_name = "/ekew_i_driver/output",
        #     topic_msg_type = "and_scale_ros/WeightStamped",
        #     buf_clear = True
        # )
    )

    dataset_rosbag_path = sys.argv[1]
    if not os.path.exists(dataset_rosbag_path):
        print("dataset")
        quit()

    # search all bags
    # rosbag_list = glob(os.path.join(dataset_rosbag_path, "**/*.bag"), recursive=True)
    rosbag_list = list()
    for cwd, dirs, files in os.walk(dataset_rosbag_path):
        [rosbag_list.append(os.path.join(cwd, file)) for file in files if os.path.splitext(file)[1] == ".bag"]
    print("find {} bags".format(len(rosbag_list)))
    
    all_data = dict()

    read_msg_list = list()
    for key in target_topics.keys():
        read_msg_list.append(target_topics[key]["topic_name"])

    # collect first topic
    for rosbag_path in rosbag_list:
        temp_data = dict()
        in_bag = rosbag.Bag(rosbag_path)
        # collect topic
        for topic, msg, t in in_bag.read_messages(topics=read_msg_list):
            for key in target_topics.keys():
                if topic == target_topics[key]["topic_name"] and key not in temp_data.keys():
                    temp_data[key] = convert_topic(msg, target_topics[key]["topic_msg_type"])
            if target_topics.keys() == temp_data.keys():
                break
        for key in temp_data.keys():
            if key not in all_data.keys():
                all_data[key] = [temp_data[key]]
            else:
                all_data[key].append(temp_data[key])
    for key in temp_data.keys():
        # print("key:{}, shape:{}".format(key, np.array(all_data[key]).shape))
        all_data[key] = np.average(np.array(all_data[key]), axis=0)

    out_path = dataset_rosbag_path+"_first.npy"
    np.save(out_path, all_data, allow_pickle=True, fix_imports=True)
    
    print('\nDone!')


if __name__ == '__main__':
    main()
