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

from algos.MRSSM.MRSSM.algo import build_RSSM
from algos.MRSSM.MRSSM.train import get_dataset_loader
from utils.evaluation.estimate_states import get_episode_data
from utils.evaluation.visualize_utils import get_pca_model, tensor2np, get_xy, get_xyz, reverse_image_observation

from ros_rssm.srv import *
import rospy

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




    def PredictPosition_RSSM(self, req):
        print("i="+str(self.i))
        self.i += 1
        print("Returning [%s + %s = %s]"%(req.a, req.b, (req.a + req.b)))
        # return AddTwoIntsResponse(req.a + req.b)

        observations_seq = dict(image_hsr_256 = observations_target["image_hsr_256"][t:t+1])
        state = self.model.estimate_state_online(observations_seq, actions[t:t+1], past_state, past_belief)

        past_belief, past_state = state["beliefs"][0], state["posterior_states"][0]
        locandscale = self.model.pose_poredict_model(past_belief)
        self.pose_predict_loc.append(tensor2np(locandscale["loc"]))
        self.pose_predict_scale.append(tensor2np(locandscale["scale"]))

        resp = SendRssmPredictPositionRespose()

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
        rospy.init_node('PredictPosition_RSSM_server')
        s = rospy.Service('PredictPosition_RSSM', SendRssmPredictPosition, self.PredictPosition_RSSM)
        print("Ready to PredictPosition_RSSM.")
        rospy.spin()

if __name__ == "__main__":
    
    test = RSSM_ros()
    test.PredictPosition_RSSM_server()