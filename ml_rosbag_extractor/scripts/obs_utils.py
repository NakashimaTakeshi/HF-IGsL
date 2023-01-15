#!/usr/bin/env python
# -*- coding: utf-8 -*-
# import sys
# sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
import math
import importlib
import rospy
import torch
import numpy as np
import cv2
import torchaudio

from topic_utils import *


def image_crop_resize(img, options):
    if options["clip_mode"] == 1:
        img = img[options["clip_width"][0]:options["clip_width"][1], options["clip_height"][0]:options["clip_height"][1]]
    elif options["clip_mode"] == 2:
        w = options["clip_width"]
        center_h = options["clip_height_center"]
        center_w = options["clip_width_center"]
        img = img[int(center_h-w/2):int(center_h+w/2),int(center_w-w/2):int(center_w+w/2)]
    if options["resize"][0] != None and options["resize"][0] != None:
        img = cv2.resize(img, options["resize"])
    return img

def image_rgb2bgr(image):
    return image.transpose(2, 0, 1)

class ObservationMaker():
    
    _topic_cfg: dict
    _buffer: dict
    _subscriber_dict: dict
    
    def __init__(self, topic_cfg: dict):
        self._topic_cfg = topic_cfg
        self._buffer = dict()
        self._subscriber_dict = dict()
        self._wait_time = 3.0
        
        self._all_topic_checker()
        self.preprocess_module = ObservationPreprocess(topic_cfg, "cpu")
        self.init_buffer()
        self._all_topic_subscriber()
        
    def init_buffer(self):
        for topic_key in self._topic_cfg.keys():
            if self._topic_cfg[topic_key]["callback_option"]["buffer_type"] == "variable":
                self._buffer[topic_key] = None
            elif self._topic_cfg[topic_key]["callback_option"]["buffer_type"] == "list":
                self._buffer[topic_key] = list()

    def _all_topic_subscriber(self):
        for topic_key in self._topic_cfg.keys():
            callback_options = self._topic_cfg[topic_key]["callback_option"]
            topic_msg_type = self._topic_cfg[topic_key]["topic_msg_type"]
            data_class = self._topic_cfg[topic_key]["topic_msg_imp"]
            buffer_mode = callback_options["buffer_mode"]
            if callback_options["buffer_type"] == "list":
                buffer_length = callback_options["buffer_length"]
            else:
                buffer_length = None
            _subscriber_kwargs = dict(
                name=self._topic_cfg[topic_key]["topic_name"],
                data_class=data_class,
                callback=self._callback_process,
                callback_args=(topic_key, topic_msg_type, buffer_mode, buffer_length),
                queue_size=1,
            )
            self._subscriber_dict[topic_key] = rospy.Subscriber(**_subscriber_kwargs)

    def _callback_process(self, msg, args):
        (topic_key, topic_msg_type, buffer_mode, buffer_length) = args
        data = self.message_converter(msg, topic_msg_type)
        if buffer_mode == "swap":
            self._buffer[topic_key] = data
        elif buffer_mode == "append":
            self._buffer[topic_key].append(data)
            self._buffer[topic_key][-buffer_length:]
        elif buffer_mode == "extend":
            self._buffer[topic_key].extend(data)
            self._buffer[topic_key][-buffer_length:]
        else:
            raise NotImplementedError("please check buffer mode : {}".format(topic_key))

    @staticmethod
    def message_converter(msg, msg_type):
        if msg_type == "sensor_msgs/Image":
            data = image_raw_msg2opencv(msg)
        elif msg_type == "sensor_msgs/CompressedImage":
            data = image_cmp_msg2opencv(msg)
        elif msg_type == "audio_common_msgs/AudioData":
            data = np.frombuffer(msg.data, dtype=np.float32)
        elif msg_type == "sensor_msgs/JointState":
            data = jointstate_converter(msg)["position"]
        elif msg_type == "geometry_msgs/PoseStamped":
            data = posestamped_converter(msg)["pose"]
        else:
            print("Error : please check the type of topic")
            raise NotImplementedError()
        return data
    
    # ----------------------------------------------------- # 
    #                   Topic Checker                       #
    # ----------------------------------------------------- # 
    def _all_topic_checker(self):
        # check the topic is exitst
        for topic_key in self._topic_cfg.keys():
            self._single_topic_checker(topic_key)

    def _single_topic_checker(self, topic_key: str):
        wait4msg_kwargs = dict(
            topic=self._topic_cfg[topic_key]["topic_name"],
            topic_type=self._topic_cfg[topic_key]["topic_msg_imp"],
            timeout=self._wait_time
        )
        try:
            data = rospy.wait_for_message(**wait4msg_kwargs)
        except rospy.ROSException:
            raise TimeoutError("[TopicChecker] The topic {} didn't response".format(self._topic_cfg[topic_key]["topic_name"]))
        if "topic_checker_option" in self._topic_cfg[topic_key].keys():
            data = self.message_converter(data, self._topic_cfg[topic_key]["topic_msg_type"])
            if self._topic_data_cheker(data, self._topic_cfg[topic_key]["topic_checker_option"]["check_type"]) == False:
                raise ValueError("[TopicChecker] Topic have some problems : {}".format(self._topic_cfg[topic_key]["topic_name"]))
        print("[TopicChecker] OK : {}".format(self._topic_cfg[topic_key]["topic_name"]))
    
    def _topic_data_cheker(self, data, check_option: list):
        result = True
        if "allzero" in check_option:
            if np.all(data == 0):
                print("[TopicChecker] data is all zero")
                result = False
        return result
    # ----------------------------------------------------- #
    
    def make_raw_observation(self):
        observation = dict()
        for topic_key in self._topic_cfg:
            observation[topic_key] = self._buffer[topic_key]
        return observation
    
    def __call__(self):
        observation = self.make_raw_observation() #create raw obs
        observation = self.preprocess_module(observation) #preprocess obs
        # observation = self.normalization_module(observation) #normalization obs
        return observation


class ObservationPreprocess():
    def __init__(self, topic_cfg: dict, device: str):
        self._topic_cfg = topic_cfg
        self.device = torch.device(device)
        self.init_modules()
        
    def init_modules(self):
        # mainly init torchaudio
        self.init_audio()
        
    def image_preprocess(self, image, option):
        image = image_crop_resize(image, option)
        # option
        # image = image_rgb2bgr(image)
        image = torch.tensor(image, dtype=torch.float32, device=self.device)
        # image_preprocess(image, bit_depth=bit_depth)
        return image
        
    def init_audio(self):
        # This is sound parameter for converting wav to mel-spectrogram
        self.sr = 16000
        self.fft_size = 1024
        self.frame_period = 5  # ms
        self.target_hz = 10
        self.n_mels = 128
        self.hop_length = int(0.001 * self.sr * self.frame_period)
        self.frame_num = int((1 / self.target_hz) / (0.001 * self.frame_period))
        self.sound_length = int(self.sr / self.target_hz)
        self.top_db = 80.0
        self.multiplier = 10.0
        self.amin = 1e-10
        self.ref_value = np.max
        self.trans_mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=self.fft_size,
            win_length=None,
            hop_length=self.hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            onesided=True,
            n_mels=self.n_mels,
        ).to(device=self.device)
        rospy.loginfo("audio initialized")
    
    def sound_preprocess(self, wav):
        # option
        wav = float32_to_pcm(wav)
        wav = torch.FloatTensor(wav, device=self.device)
        mlsp_power = self.trans_mel(wav)
        ref_value = mlsp_power.max(dim=1)[0].max(dim=0)[0]
        mlsp = torchaudio.functional.amplitude_to_DB(mlsp_power, self.multiplier, self.amin, math.log10(max(self.amin, ref_value)), self.top_db)
        mlsp = mlsp.abs().float().div_(80).narrow(1, 0, self.frame_num)
        return mlsp
    
    def __call__(self, obs: dict):
        for key in obs.keys():
            if "image" in key and "image_option" in self._topic_cfg[key].keys():
                obs[key] = self.image_preprocess(obs[key], self._topic_cfg[key]["image_option"])
            elif "sound" in key:
                obs[key] = self.sound_preprocess(obs[key])
            else:
                obs[key] = numpy2tensor(obs[key])
        return obs
