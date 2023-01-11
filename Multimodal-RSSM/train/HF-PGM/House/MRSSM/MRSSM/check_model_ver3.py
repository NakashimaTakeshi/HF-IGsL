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

module_path = os.path.join(Path().resolve(), '../../../../..')
sys.path.append(module_path)

#pythonpathの追加
os.environ['PYTHONPATH'] = module_path

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

    pose_array[0][3]=quaternion_to_euler_zyx(action[3:7])
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
model_paths

model = build_RSSM(cfg, device)
model_path = model_paths[model_idx]
model.load_model(model_path)
model.eval()

#load data
if data=="train":
    D = get_dataset_loader(cfg, cwd, device, cfg.train.train_data_path)
elif data=="validation":
    D = get_dataset_loader(cfg, cwd, device, cfg.train.validation_data_path)
else:
    raise NotImplementedError

# D_test = get_dataset_loader(cfg, cwd, device, cfg.train.test_data_path)


# # Reconstruction
epi_idx = 2
crop_idx = 0
observations, actions, rewards, nonterminals = get_episode_data(D, epi_idx=epi_idx, crop_idx=crop_idx)

for name in observations.keys():
    if "image_horizon" in name:
        if "bin" in name:
            image_name_horizon_bin = name
            print(name)
        else:
            image_name_horizon = name
            print(name)
            
    if "image_vertical" in name:
        if "bin" in name:
            image_name_vertical_bin = name
            print(name)
        else:
            image_name_vertical = name
            print(name)
    


size = (256,256)

# load states
state_path = model_path.replace("models", "states_models").replace(".pth", ".npy")
print("state_path:", state_path)

states_np = np.load(state_path, allow_pickle=True).item()
print("-- dataset --")
for key in states_np.keys():
    print(key)

print("-- key of states --")
print(states_np[key].keys())

'''
ht = [states_np[key]["beliefs"] for key in states_np.keys()]
ht = np.vstack(ht)
pca_ht = get_pca_model(ht, 3)
pca_ht_2d = get_pca_model(ht, 2)
st_q = [states_np[key]["posterior_means"] for key in states_np.keys()]
st_q = np.vstack(st_q)
pca_st_q = get_pca_model(st_q, 3)
pca_st_q_2d = get_pca_model(st_q, 2)
'''
#再構成？
observations_target = model._clip_obs(observations, idx_start=1)
print("observations_target.shape={},actions.shape={}".format(observations_target['image_hsr_256'].shape,actions[:-1].shape))
# state = model.estimate_state(observations_target, actions[:-1], rewards, nonterminals[:-1])
# recon = model.observation_model(h_t=state["beliefs"], s_t=state["posterior_means"])
# recon["state"]={"beliefs":state["beliefs"],"posterior_means":state["posterior_means"]}

pose_predict_loc = []
pose_predict_scale = []

past_belief, past_state = torch.zeros(1, model.cfg.rssm.belief_size, device=model.cfg.main.device), torch.zeros(1, model.cfg.rssm.state_size, device=model.cfg.main.device)

for t in range(1000):
    observations_seq = dict(image_hsr_256 = observations_target["image_hsr_256"][t:t+1])
    state = model.estimate_state_online(observations_seq, actions[t:t+1], past_state, past_belief)

    past_belief, past_state = state["beliefs"][0], state["posterior_states"][0]
    locandscale = model.pose_poredict_model(past_belief)
    pose_predict_loc.append(tensor2np(locandscale["loc"]))
    pose_predict_scale.append(tensor2np(locandscale["scale"]))
    print(t)

pose_predict_loc = np.stack(pose_predict_loc)
#位置尤度計算





#部屋ラベル分け
'''
room_label = []
for t in range(len(actions)-1):
    room_label.append(room_clustering(actions.detach().cpu().numpy()[t, 0]))
room_label = np.stack(room_label)
print(room_label)
'''

#イマジネーション
'''
t_imag_start = 30

h_t_img = [state["beliefs"][t_imag_start]]
s_t_img = [state["posterior_means"][t_imag_start]]

t_max = len(actions)
horizon_imagination = t_max-t_imag_start
for t in range(horizon_imagination):
    belief, _, prior_mean, _ = model.transition_model(s_t_img[t], actions[t_imag_start+t].unsqueeze(dim=0), h_t_img[t], det=True)
    h_t_img.append(belief.squeeze(dim=0))
    s_t_img.append(prior_mean.squeeze(dim=0))

h_t_img = torch.stack(h_t_img)
s_t_img = torch.stack(s_t_img)

recon_imag = model.observation_model(h_t=h_t_img, s_t=s_t_img)
recon_imag["state"]={"beliefs":h_t_img,"posterior_means":s_t_img}
'''

#潜在空間表現(imgnation)(PCA)
'''
ht_img = recon_imag["state"]["beliefs"]
shape = ht_img.shape[-1]
feat = tensor2np(ht_img).reshape(-1, shape)
feat_pca = pca_ht.transform(feat)
hx_img, hy_img, hz_img = get_xyz(feat_pca)

st_q_img = recon_imag["state"]["posterior_means"]
shape = st_q_img.shape[-1]
feat = tensor2np(st_q_img).reshape(-1, shape)
feat_pca = pca_st_q.transform(feat)
sx_img, sy_img, sz_img = get_xyz(feat_pca)
'''
