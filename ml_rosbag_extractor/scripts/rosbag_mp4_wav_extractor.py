#!/usr/bin/env python2
import os
import sys
import numpy as np
import rosbag
import warnings

from record_utils import *
from topic_utils import *


target_topics = dict(
    image_horizon=dict(
        topic_name = "/camera_side/color/image_raw/compressed",
        topic_msg_type = "sensor_msgs/CompressedImage",
        topic_hz = 30.0,
    ),
    image_vertical_high_resolution=dict(
        topic_name = "/usb_cam/image_raw/compressed",
        topic_msg_type = "sensor_msgs/CompressedImage",
        topic_hz = 30.0,
    ),
    sound=dict(
        topic_name = "/audio/audio",
        topic_msg_type = "audio_common_msgs/AudioData",
        sound_info_topic = "/audio/audio_info",
    ),
)

def collect_topics(rosbag, collect_topic, topic_msg_type):
    t_s = rosbag.get_start_time()
    t_e = rosbag.get_end_time()
    collect_topic_list = []
    for topic, msg, t in rosbag.read_messages(topics=collect_topic):
        percentage = (t.to_sec() - t_s) / (t_e - t_s)
        sys.stdout.write('\r collect {0} : {1:4.2f}%'.format(collect_topic, 100 * percentage))
        t = t.to_sec() - t_s
        data = topic_preprocess(msg, topic_msg_type)
        if topic_msg_type == "audio_common_msgs/AudioData":
            collect_topic_list.extend(data)
        else:
            collect_topic_list.append(data)
    sys.stdout.flush()
    return collect_topic_list


def topic_preprocess(msg, topic_msg_type):
    if topic_msg_type == "sensor_msgs/Image":
        data = image_raw_msg2opencv(msg)
    elif topic_msg_type == "sensor_msgs/CompressedImage":
        data = image_cmp_msg2opencv(msg)
    elif topic_msg_type == "audio_common_msgs/AudioData":
        data = np.frombuffer(msg.data, dtype=np.float32)
    elif topic_msg_type == "sensor_msgs/JointState":
        data = jointstate_converter(msg)["position"]
    elif topic_msg_type == "geometry_msgs/PoseStamped":
        data = posestamped_converter(msg)["pose"]
    else:
        raise NotImplementedError("please check the type of topic")
    return data


def create_mp4(image_list, output_path, frame_rate):
    size = (image_list[0].shape[1], image_list[0].shape[0])
    frame_rate = frame_rate
    fourcc = cv2.VideoWriter_fourcc('m','p','4', 'v')
    save = cv2.VideoWriter(output_path+'.mp4', fourcc, frame_rate, size)
    for i in range(len(image_list)):
        save.write(image_list[i])
    save.release()
    cv2.destroyAllWindows()
    del save
    

def convine_mp4_wav(video_path, wav_path, out_path, overwite=True):
    import ffmpeg
    instream_v = ffmpeg.input(video_path)
    instream_a = ffmpeg.input(wav_path)
    stream = ffmpeg.output(instream_v, instream_a, out_path, vcodec="copy", acodec="aac")
    ffmpeg.run(stream, overwrite_output=overwite, quiet=True)


def create_wav(wav_data, output_path, sr):
    import soundfile as sf
    _format = "WAV"
    subtype = 'FLOAT'
    sf.write(output_path+".wav", wav_data, sr, format=_format, subtype=subtype)


if __name__ == '__main__':
    bag_path = sys.argv[1]
    in_bag = rosbag.Bag(bag_path)
    
    out_dir_name = os.path.splitext(bag_path)[0]
    if not os.path.exists(out_dir_name):
        os.makedirs(out_dir_name)
    
    # collect topic
    for key in target_topics.keys():
        output_path = os.path.join(out_dir_name, out_dir_name+'_'+key)
        data_list = collect_topics(in_bag, target_topics[key]["topic_name"], target_topics[key]["topic_msg_type"])
        if "image" in key:
            create_mp4(data_list, output_path, target_topics[key]["topic_hz"])
        elif "sound" in key:
            audio_info = get_audio_info(in_bag, target_topics[key]["sound_info_topic"])
            create_wav(data_list, output_path, audio_info["sampling_rate"])
        else:
            raise NotImplementedError()
        del data_list
