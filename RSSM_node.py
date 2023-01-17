#!/usr/bin/env python3
# coding: utf-8
import sys
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import hydra
import torch
import matplotlib.pyplot as plt
from hydra import initialize, compose
import glob
from scipy.spatial.transform import Rotation as R
import cv2

from algos.MRSSM.MRSSM.algo import build_RSSM
from algos.MRSSM.MRSSM.train import get_dataset_loader
from utils.evaluation.estimate_states import get_episode_data
from utils.evaluation.visualize_utils import get_pca_model, tensor2np, get_xy, get_xyz, reverse_image_observation

from ros_rssm.srv import *
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import PoseWithCovarianceStamped

class RSSM_ros():
    def __init__(self):
        # 勾配を計算しない
        torch.set_grad_enabled(False)

        # #パラーメーター設定
        path_name = "HF-PGM_Predicter_0-seed_0/2022-12-15/run_12"
        model_idx = 5
        cfg_device = "cuda:0"

        # model_folder = "/root/TurtleBot3/catkin_ws/src/ros_rssm/scripts/results/HF-PGM_Predicter_0-seed_0/2022-12-15/run_12"
        model_folder = os.path.join("results", path_name)



        with initialize(config_path=model_folder):
            cfg = compose(config_name="hydra_config")

        # #仮の修正
        cfg.main.wandb=False

        cfg.main.device = cfg_device
        # # cfg.train.n_episode = 100
        print(' ' * 26 + 'Options')
        for k, v in cfg.items():
            print(' ' * 26 + k + ': ' + str(v))

        device = torch.device(cfg.main.device)

        # # # Load Model, Data and States
        model_paths = glob.glob(os.path.join(model_folder, '*.pth'))
        print(model_paths)
        print("model_pathes: ")

        self.model = build_RSSM(cfg, device)
        model_path = model_paths[model_idx]
        self.model.load_model(model_path)
        self.model.eval()

        # # load states
        state_path = model_path.replace("models", "states_models").replace(".pth", ".npy")
        print("state_path:", state_path)

        states_np = np.load(state_path, allow_pickle=True).item()
        print("-- dataset --")
        for key in states_np.keys():
            print(key)

        print("-- key of states --")
        print(states_np[key].keys())

        self.pose_predict_loc = []
        self.pose_predict_scale = []
        self.past_belief = torch.zeros(1, self.model.cfg.rssm.belief_size, device=self.model.cfg.main.device)
        self.past_state = torch.zeros(1, self.model.cfg.rssm.state_size, device=self.model.cfg.main.device)
        

    def ros_init_(self):
        rospy.init_node('PredictPosition_RSSM_server')
        self.pose_subscriber()
        self.image_subscriber()
        self.PredictPosition_RSSM_server()
        print("Ready to PredictPosition_RSSM.")
        rospy.spin()

    def pose_subscriber(self):
        rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.callback_pose)

    def image_subscriber(self):
        rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback_image)

    def callback_image(self, msg):
        bridge = CvBridge()
        img_cv2 = bridge.imgmsg_to_cv2(msg, "bgr8")
        img_cut = img_cv2[ :, 80:560]
        self.img = cv2.resize(img_cut,(256, 256))
        # print(self.img.shape)
        # print(self.pose)

    def callback_pose(self, msg):
        self.pose = self.posewithcovariancestamped_converter(msg)


    def posewithcovariancestamped_converter(self, msg):
        pose_list = self.pose_converter(msg.pose.pose)
        pose_list_oira = self.quaternion2euler_numpy(pose_list[3], pose_list[4], pose_list[5], pose_list[6])
        pose_data = [pose_list[0], pose_list[1], np.cos(pose_list_oira[2]), np.sin(pose_list_oira[2])]
        return np.array(pose_list)
    
    def pose_converter(self, msg):
        position_list = self.geometry_msgs_vector3d_converter(msg.position)
        orientation_list = self.geometry_msgs_quaternion_converter(msg.orientation)
        return position_list + orientation_list

    def geometry_msgs_vector3d_converter(self, msg):
        return [msg.x, msg.y, msg.z]

    def geometry_msgs_quaternion_converter(self, msg):
        return [msg.x, msg.y, msg.z, msg.w]

    def quaternion2euler_numpy(self, x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.degrees(np.arctan2(t0, t1))
        t2 = +2.0 * (w * y - z * x)
        t2 = np.where(t2>+1.0,+1.0,t2)
        t2 = np.where(t2<-1.0, -1.0, t2)
        pitch_y = np.degrees(np.arcsin(t2))
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.degrees(np.arctan2(t3, t4))
        return roll_x, pitch_y, yaw_z # in radians


    def PredictPosition_RSSM(self, req):
        normalized_img = normalize_image(np2tensor(self.img.transpose(2, 0, 1)), 5).unsqueeze(0).unsqueeze(0).to(device = self.model.device)
        action = np2tensor(self.pose).unsqueeze(0).unsqueeze(0).to(device = self.model.device)

        observations_seq = dict(image_hsr_256 = normalized_img)
        state = self.model.estimate_state_online(observations_seq, action, self.past_state, self.past_belief)

        self.past_belief, self.past_state = state["beliefs"][0], state["posterior_states"][0]
        locandscale = self.model.pose_poredict_model(self.past_belief)
        self.pose_predict_loc.append(tensor2np(locandscale["loc"])[0].tolist())
        self.pose_predict_scale.append(tensor2np(locandscale["scale"])[0].tolist())
        print(self.pose_predict_loc[-1])

        resp = SendRssmPredictPositionResponse()

        resp.x_loc = self.pose_predict_loc[-1][0]
        resp.y_loc = self.pose_predict_loc[-1][1]
        resp.cos_loc = self.pose_predict_loc[-1][2]
        resp.sin_loc = self.pose_predict_loc[-1][3]
        resp.x_scale = self.pose_predict_scale[-1][0]
        resp.y_scale = self.pose_predict_scale[-1][1]
        resp.cos_scale = self.pose_predict_scale[-1][2]
        resp.sin_scale = self.pose_predict_scale[-1][3]

        return resp


    def PredictPosition_RSSM_server(self):
        s = rospy.Service('PredictPosition_RSSM', SendRssmPredictPosition, self.PredictPosition_RSSM)
        

def normalize_image(observation, bit_depth):
    # Quantise to given bit depth and centre
    observation.div_(2 ** (8 - bit_depth)).floor_().div_(2 ** bit_depth).sub_(0.5)
    # Dequantise (to approx. match likelihood of PDF of continuous images vs. PMF of discrete images)
    observation.add_(torch.rand_like(observation).div_(2 ** bit_depth))
    return observation

def np2tensor(data, dtype = torch.float32):
    if torch.is_tensor(data):
        return data
    else:
        return torch.tensor(data, dtype=dtype)

def tensor2np(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().detach().numpy().copy()
    else:
        return tensor



if __name__ == "__main__":
    
    test = RSSM_ros()
    test.ros_init_()