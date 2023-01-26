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

from algos.MRSSM.MRSSM.algo import build_RSSM
from algos.MRSSM.MRSSM.train import get_dataset_loader
from utils.evaluation.estimate_states import get_episode_data
from utils.evaluation.visualize_utils import get_pca_model, tensor2np, get_xy, get_xyz, reverse_image_observation
from utils.dist import calc_subset_states

from utils.models.pose_predict_model import PosePredictModel

# from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R


# 勾配を計算しない
torch.set_grad_enabled(False)

#パラーメーター設定
state_path = "dataset_1_test.bag20230126_204953" + ".npy"
path_name = "HF-PGM_model1-seed_0/2023-01-25/run_3"
cfg_device = "cuda:0"
#cfg_device = "cpu"
model_idx = 2
#イマジネーションスタート時刻
t_imag_start = 30
model_folder = os.path.join("results", path_name)
folder_name = os.path.join("figs_out_dir/save_folda", path_name)
size = (256, 256)


with initialize(config_path=model_folder):
    cfg = compose(config_name="hydra_config")

#仮の修正
cfg.main.wandb=False

cfg.main.device = cfg_device
device = torch.device(cfg.main.device)

# # Load Model, Data and States
model_paths = glob.glob(os.path.join(model_folder, '*.pth'))
print("model_pathes: ")
model_paths

model = build_RSSM(cfg, device)
model_path = model_paths[model_idx]
model.load_model(model_path)
model.eval()

states_np = np.load(state_path, allow_pickle=True).item()

#イマジネーション
print(states_np.keys())

def imagination(h_t, s_t, actions, step):
    """
    action : t-1 -> T-1
    """
    h_t_img = [h_t]
    s_t_img = [s_t]
    
    for t in range(step):
        belief, _, prior_mean, _ = model.transition_model(s_t_img[t], actions[t].unsqueeze(dim=0), h_t_img[t], det=True)
        h_t_img.append(belief.squeeze(dim=0))
        s_t_img.append(prior_mean.squeeze(dim=0))

    h_t_img = torch.stack(h_t_img)
    s_t_img = torch.stack(s_t_img)

    recon_imag = model.observation_model(h_t=h_t_img, s_t=s_t_img)
    recon_imag["state"]={"beliefs":h_t_img,"posterior_means":s_t_img}

    return recon_imag

def reconstruction_single_Rssm(o_t):
    normed_image = []
    for t in range(len(o_t)):
        normed_image.append(normalize_image(np2tensor(o_t[t]), 5).unsqueeze(0).to(device = model.device))
    
    o_t = torch.stack(normed_image)

    observations_target = model._clip_obs(o_t, idx_start=1)
    T = observations_target.size(0)

    actions = torch.zeros([T, 1, 1], device = model.device)
    rewards = torch.zeros([T, 1, 1], device = model.device)
    nonterminals = torch.ones([T, 1, 1], device = model.device)

    state = model.estimate_state(observations_target, actions[:-1], rewards, nonterminals[:-1])
    recon = model.observation_model(h_t=state["beliefs"], s_t=state["posterior_means"])
    recon["state"] = {"beliefs":state["beliefs"],"posterior_means":state["posterior_means"]}

    return recon


def crossmodal_recon(past_belief, past_state, x_t, step):

    actions = torch.zeros([1, 1, 1], device = model.device)
    # rewards = torch.zeros([step, 1, 1], device = model.device)
    # nonterminals = torch.ones([step, 1, 1], device = model.device)
    img_dummy = torch.zeros((1, 1, *model.cfg.env.observation_shapes["image_hsr_256"]), device = model.cfg.main.device)

    past_belief = np2tensor(past_belief).to(device = model.device)
    past_state = np2tensor(past_state).to(device = model.device)

    h_t = [past_belief]
    s_t = [past_state]

    x_t = np.array(x_t)

    print("x_t",x_t.shape)

    for t in range(step):
        observations_seq = dict(image_hsr_256 = img_dummy, Pose = np2tensor(x_t[t]).unsqueeze(0).unsqueeze(0).to(device = model.device))  # Poseは状態の推論には使用しないが、何らかの値がないとエラーが出る
        print("ht:{},st:{},at:{}".format(h_t[-1].shape, s_t[-1].shape, actions.shape))
        state = model.estimate_state_online(observations_seq, actions, s_t[-1], h_t[-1], subset_index = 2)

        h_t.append(state["beliefs"].squeeze(0))
        s_t.append(state["posterior_states"].squeeze(0))
    
    h_t = torch.stack(h_t)
    s_t = torch.stack(s_t)
    
    recon = model.observation_model(h_t=h_t, s_t=s_t)
    recon["state"] = {"beliefs":h_t,"posterior_means":s_t}

    return recon



def reconstruction_single_Rssm(o_t):
    normed_image = []
    for t in range(len(o_t)):
        normed_image.append(normalize_image(np2tensor(o_t[t]), 5).unsqueeze(0).to(device = model.device))
    
    o_t = torch.stack(normed_image)

    observations_target = model._clip_obs(o_t, idx_start=1)
    T = observations_target.size(0)

    actions = torch.zeros([T, 1, 1], device = model.device)
    rewards = torch.zeros([T, 1, 1], device = model.device)
    nonterminals = torch.ones([T, 1, 1], device = model.device)

    state = model.estimate_state(observations_target, actions[:-1], rewards, nonterminals[:-1])
    recon = model.observation_model(h_t=state["beliefs"], s_t=state["posterior_means"])
    recon["state"] = {"beliefs":state["beliefs"],"posterior_means":state["posterior_means"]}

    return recon


def normalize_image(observation, bit_depth):
    # Quantise to given bit depth and centre
    observation.div_(2 ** (8 - bit_depth)).floor_().div_(2 ** bit_depth).sub_(0.5)
    # Dequantise (to approx. match likelihood of PDF of continuous images vs. PMF of discrete images)
    observation.add_(torch.rand_like(observation).div_(2 ** bit_depth))
    return observation

def np2tensor(data, dtype=torch.float32):
    if torch.is_tensor(data):
        return data
    else:
        return torch.tensor(data, dtype=dtype)

def tensor2np(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().detach().numpy().copy()
    else:
        return tensor

def reverse_image_observation(image, bit_depth=5):    
    image = reverse_normalized_image(image, bit_depth=bit_depth).transpose(1, 2, 0)
    return image

# Postprocess an observation for storage (from float32 numpy array [-0.5, 0.5] to uint8 numpy array [0, 255])
def reverse_normalized_image(observation, bit_depth=5):
  return np.clip(np.floor((observation + 0.5) * 2 ** bit_depth) * 2 ** (8 - bit_depth), 0, 2 ** 8 - 1).astype(np.uint8)








#model2

# recon = crossmodal_recon(states_np['past_belief'][5], states_np['past_posterior_states'][5], states_np['pose_t-1'][7:17], 10)

# im = recon["image_hsr_256"]["loc"][5, 0][[2, 1, 0]].detach().cpu().numpy()

im = reverse_image_observation(im)
im = cv2.resize(im, size, interpolation=cv2.INTER_LINEAR)
plt.imshow(im)
plt.savefig("test.png")
