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
from std_msgs import Image
from geometry_msgs import PoseWithCovarianceStamped

class RSSM_ros():
    def __init__(self):
        # 勾配を計算しない
        torch.set_grad_enabled(False)

        #パラーメーター設定
        path_name = "HF-PGM_Predicter_0-seed_0/2022-12-15/run_12"
        model_idx = 5
        cfg_device = "cuda:0"
        #cfg_device = "cpu"
        data="validation"

        cwd = "."
        model_folder = os.path.join("results", path_name)
        folder_name = os.path.join("figs_out_dir/save_folda", path_name)

        with initialize(config_path=model_folder):
            cfg = compose(config_name="hydra_config")

        #仮の修正
        cfg.main.wandb=False

        cfg.main.device = cfg_device
        # cfg.train.n_episode = 100
        print(' ' * 26 + 'Options')
        for k, v in cfg.items():
            print(' ' * 26 + k + ': ' + str(v))

        device = torch.device(cfg.main.device)
        pass
        # # Load Model, Data and States
        model_paths = glob.glob(os.path.join(model_folder, '*.pth'))
        print("model_pathes: ")

        self.model = build_RSSM(cfg, device)
        model_path = model_paths[model_idx]
        self.model.load_model(model_path)
        self.model.eval()

        # load states
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

    def ros_init(self):
        rospy.init_node('PredictPosition_RSSM_server')
        self.pose_subscriber()
        self.img_subscriber()
        self.PredictPosition_RSSM_server()
        print("Ready to PredictPosition_RSSM.")
        rospy.spin()

    def img_subscriber(self):
        rospy.Subscriber("chatter", Image, self.img_callback)

    def pose_subscriber(self):
        rospy.Subscriber("chatter", PoseWithCovarianceStamped, self.pose_callback)

    def img_callback(self, msg):
        bridge = CvBridge()
        cv_array = bridge.imgmsg_to_cv2(msg)
        self.img = cv2.resize(cv_array[80:560,:],(256,256))

    def pose_callback(self, msg):
        self.pose = posewithcovariancestamped_converter(msg)

    def PredictPosition_RSSM(self, req):
        normaziimg = 
        observations_seq = dict(image_hsr_256 = observations_target["image_hsr_256"][t:t+1])
        state = self.model.estimate_state_online(observations_seq, actions[t:t+1], past_state, past_belief)

        self.past_belief, self.past_state = state["beliefs"][0], state["posterior_states"][0]
        locandscale = self.model.pose_poredict_model(self.past_belief)
        self.pose_predict_loc.append(tensor2np(locandscale["loc"]).tolist()[0])
        self.pose_predict_scale.append(tensor2np(locandscale["scale"]).tolit()[0])

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
        
def quaternion2euler_numpy(x, y, z, w):
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


def posewithcovariancestamped_converter(msg):
    pose_list = pose_converter(msg.pose.pose)
    pose_list_oira = quaternion2euler_numpy(*pose_list[3:7])
    pose_data = [pose_list[0], pose_list[1], np.cos(pose_list_oira[2]), np.sin(pose_list_oira[2])]
    return np.array(pose_data)

def pose_converter(msg):
    position_list = geometry_msgs_vector3d_converter(msg.position)
    orientation_list = geometry_msgs_quaternion_converter(msg.orientation)
    return position_list + orientation_list

def geometry_msgs_vector3d_converter(msg):
    return [msg.x, msg.y, msg.z]

def geometry_msgs_quaternion_converter(msg):
    return [msg.x, msg.y, msg.z, msg.w]


if __name__ == "__main__":
    
    test = RSSM_ros()
    test.ros_init_()