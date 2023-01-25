#!/usr/bin/env python
# coding: utf-8
import sys
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from omegaconf import OmegaConf
import cv2
import mpl_toolkits
import glob
import math
from sklearn.manifold import TSNE
import datetime
import matplotlib.patches as pat

module_path = os.path.join(Path().resolve(), '../../../../..')
sys.path.append(module_path)

#pythonpathの追加
os.environ['PYTHONPATH'] = module_path
sys.path.append(os.path.join(Path().resolve(), '../Multimodal-RSSM'))

from algos.MRSSM.MRSSM.algo import build_RSSM
from algos.MRSSM.MRSSM.train import get_dataset_loader
from utils.evaluation.estimate_states import get_episode_data
from utils.evaluation.visualize_utils import get_pca_model, tensor2np, get_xy, get_xyz, reverse_image_observation
from utils.dist import calc_subset_states

from utils.models.pose_predict_model import PosePredictModel

# from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R

def quaternion_to_euler_zyx(q):
    r = R.from_quat([q[0], q[1], q[2], q[3]])
    return r.as_euler('xyz', degrees=True)[2]

def display_pose(action):
    #print("\n")
    #print(self.observations["Pose"][-1])
    pose_array=np.array([[0,0,0,0]],dtype=float)
    
    pose_array[0][0]=action[0]*20+190
    pose_array[0][1]=action[1]*(-20)+115

    if len(action) == 7:
        pose_array[0][3]=quaternion_to_euler_zyx(action[3:7])
    elif len(action) == 4:
        pose_array[0][3] = math.degrees(math.atan(action[3]/action[2]))
    return pose_array

def room_clustering(position):
    x = position[0]
    y = position[1]

    if x <= -2.5:
        if y > -1.5:
            room = 0
        else:
            room = 1
    elif x > -2.5 and x <=4.75:
        if y > 1.3:
            room = 2
        else:
            room = 3
    else:
        if y > -1.3:
            room = 4
        else:
            room = 5

    return room

#パラーメーター設定
import glob

file_dir = "eval_data/amcl/dataset1"
files = glob.glob("./"+ file_dir +"/*")
data_np = np.load(files[0], allow_pickle=True).item()
num = len(data_np["pose_t-1"])-1
print(num)

culc_data = np.zeros((len(files), num))
print("Num=",num)


for file in range(len(files)):
    # file_path = os.path.join(file_dir, files[file]) 
    data_np = np.load(files[file], allow_pickle=True).item()
    # print(files[file])
    # print(data_np.keys())
    print(files[file])
    for i in range(num):
        data_pose = data_np["pose_t-1"][i+1]
        data_grand_pose = data_np["grand_pose_t"][i]
        # print("{}-{}={}\n".format(data_grand_pose,data_pose,data_grand_pose-data_pose))

        culc_data[file][i] = round(np.sqrt(np.mean((data_grand_pose - data_pose) ** 2)),2)
    plt.plot(np.arange(num), culc_data[file], alpha=0.15,  lw=1)
culc_data.mean(axis=0)



plt.plot(np.arange(len(culc_data.mean(axis=0))),culc_data.mean(axis=0), color = "red", lw=3)

plt.savefig("test.png")
    

    # #部屋ラベル分け
    # room_label = []
    # for t in range(len(actions)-1):
    #     room_label.append(room_clustering(observations_target['Pose'].detach().cpu().numpy()[t, 0]))
    # room_label = np.stack(room_label)
    # print(room_label)
