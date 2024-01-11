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
# import cv2
# import mpl_toolkits
import glob
import math
# from sklearn.manifold import TSNE
import datetime
# import matplotlib.patches as pat

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

# Load Config
train_log_dir = "HF-PGM_model1-seed_0/2023-11-26/run_2"
# train_log_dir = "HF-PGM_model2-seed_0/2023-11-26/run_1"
mun_iteration = "_3000.pth"

data="train"
# data="validation"
cwd = "."
model_dir = os.path.join("results", train_log_dir)
output_dir = os.path.join("figs_out_dir/save_folda", train_log_dir)

with initialize(config_path=model_dir):
    cfg = compose(config_name="hydra_config")
cfg.main.wandb=False

print(' ' * 26 + 'Options')
for k, v in cfg.items():
    print(' ' * 26 + k + ': ' + str(v))

# Load Model, States and Data
model_paths = glob.glob(os.path.join(model_dir, '*.pth'))
model_path = [item for item in model_paths if mun_iteration in item][0]
print(f"model_pathes: {model_paths}")

device = torch.device(cfg.main.device)
model = build_RSSM(cfg, device)
model.load_model(model_path)
model.eval()

if 'model1'in model_path.lower():
    model_type="model1"
elif 'model2'in model_path.lower():
    model_type="model2"
else:
    raise NotImplementedError

#load data
if data=="train":
    D = get_dataset_loader(cfg, cwd, device, cfg.train.train_data_path)
elif data=="validation":
    D = get_dataset_loader(cfg, cwd, device, cfg.train.validation_data_path)
else:
    raise NotImplementedError

def cell_norm_online(cells, positions, current_cell_mat, current_num_visits,num_discretize):
    n_episode=1

    num_cells = np.shape(cells)[2]
    len_episode = np.shape(cells)[0]
    cell_mat = np.zeros((num_discretize[0], num_discretize[1], num_discretize[2], num_cells))
    num_visits = np.zeros((num_discretize[0], num_discretize[1], num_discretize[2], 1))

    for epi in range(n_episode):
        for ii in range(len_episode):
            position = positions[ii, epi, :]
            cell_mat[int(position[0]), int(position[1]), int(position[2]) , :] += cells[ii, epi, :].detach().cpu().numpy()
            num_visits[int(position[0]), int(position[1]), int(position[2])] += 1
            # cell_mat[env][position, :] += cells[env, :, ii]
        try:
            new_cell_mat = cell_mat + current_cell_mat
        except:
            new_cell_mat = cell_mat
        
        new_num_visits = num_visits + current_num_visits        

    return new_cell_mat, new_num_visits

n_episord = 9
range_discretize = [(-6,6),(-5,5),(-np.pi,np.pi)]
# num_discretize = [40,32,2]
num_discretize = [48,40,2]
# num_discretize = [24,20,2]

ht_all = np.zeros(tuple(num_discretize)+ (200,))
st_all = np.zeros(tuple(num_discretize)+ (30,))
num_visits = np.zeros(tuple(num_discretize)+ (1,))

for epi_idx in range(n_episord):
    observations, actions, rewards, nonterminals = get_episode_data(D, epi_idx = epi_idx)
    observations_target = model._clip_obs(observations, idx_start=1)

    if model_type == "model1":
        state = model.estimate_state(observations_target, actions[:-1], rewards, nonterminals[:-1], det=True)
        positions = actions[0:-1]
    elif model_type == "model2":
        state = model.estimate_state(observations_target, actions[:-1], rewards, nonterminals[:-1], subset_index = 1,det=True)
        positions = observations_target["Pose"]

    def discretize_pose(positions, range_discretize, num_discretize):
        
        def torch_digitize(tensor, bins):
            indices = torch.searchsorted(bins, tensor, right=True)
            return indices

        x_pos = positions[:,:,0]
        y_pos = positions[:,:,1]        
        theata_angle = torch.atan2(positions[:,:,3],positions[:,:,2])
        
        discretized_x = torch_digitize(x_pos, torch.from_numpy(np.linspace(range_discretize[0][0], range_discretize[0][1], num_discretize[0] + 1)[1:-1]).to(x_pos.device))
        discretized_y = torch_digitize(y_pos, torch.from_numpy(np.linspace(range_discretize[1][0], range_discretize[1][1], num_discretize[1] + 1)[1:-1]).to(y_pos.device))
        discretized_theta = torch_digitize(theata_angle, torch.from_numpy(np.linspace(range_discretize[2][0], range_discretize[2][1], num_discretize[2] + 1)[1:-1]).to(theata_angle.device))
        
        return torch.concatenate([discretized_x, discretized_y, discretized_theta], dim=1)

    discretized_positons = discretize_pose(positions.detach(), range_discretize, num_discretize)

    ht_all, num_visits = cell_norm_online(state["beliefs"], discretized_positons.unsqueeze(dim=1), ht_all, num_visits,num_discretize)
    st_all, _ = cell_norm_online(state["posterior_means"], discretized_positons.unsqueeze(dim=1), st_all, num_visits,num_discretize)

num_visits_safe = np.where(num_visits == 0, 1, num_visits)
ht_all_norm = np.where(num_visits!=0, ht_all/num_visits_safe, 0.0)
st_all_norm = np.where(num_visits!=0, st_all/num_visits_safe, 0.0)

def cell_plot_prepare(cell_,direction_state=None):
    if direction_state == None:
        cell_reshaped = np.sum(cell_, axis=2).T[::-1, :]/np.shape(cell_)[2]
        # cell_reshaped = np.sum(cell_, axis=2).T[::-1, :]
    else:
        cell_reshaped = cell_[:,:,direction_state].T[::-1, :]  

    return cell_reshaped

def square_plot(cell, save_name="test", maxmin=False, lims=False, direction_state=None, cmap='jet', interpolation_method=None):
    n = np.shape(cell)[-1]  # number of cells we have
    wid = np.ceil(np.sqrt(n))  # dim for subplots
    wid0, wid1 = np.ceil(n/wid), wid

    if wid0 == 1 and wid1 == 1:
        wid0, wid1 = 2, 2

    fig, axs = plt.subplots(int(wid0), int(wid1),figsize=(12, 10))

    for row in axs:
        for ax in row:
            ax.axis('off') 
            ax.set_yticks([])
            ax.set_xticks([])

    for grid in range(n):
        cell_ = cell[:,:,:, grid]
        i = int(grid % int(wid1))
        j = int(grid // int(wid1))

        cell_reshaped = cell_plot_prepare(cell_,direction_state)

        if lims:
            im = axs[j][i].imshow(cell_reshaped, cmap=cmap, interpolation=interpolation_method, vmin=lims[0],vmax=lims[1])
        else:
            im = axs[j][i].imshow(cell_reshaped, cmap=cmap,interpolation=interpolation_method)


        axs[j][i].axis('on') 

        if maxmin:
            maxi = max(cell_)
            mini = min(cell_)
            axs[j][i].set_title("{:.2f},{:.2f}".format(mini, maxi), {'fontsize': 10})

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.03, hspace=0.0)        

    if lims:
        Bbox = axs[0][0].get_position().bounds
        ratio = 0.15
        offset = -0.2 * Bbox[3]
        cax = plt.axes((Bbox[0]+ offset , Bbox[1], Bbox[3]* ratio, Bbox[3]))
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(plt.Normalize(vmin=lims[0],vmax=lims[1]),cmap=cmap),
            ax=axs,
            cax=cax,
            )
        if wid0 == 2 and wid1 == 2:
            cbar.ax.tick_params(labelsize=10)
        else:
            cbar.ax.tick_params(labelsize=3)
        cbar.ax.yaxis.set_ticks_position('left')

    fig.savefig("./" + save_name + ".pdf", bbox_inches='tight')

    plt.close('all')

now = datetime.datetime.now()
sum_orientatation = True
if sum_orientatation:
    save_name_h = "ht_{}_state({}-{})_{}".format(model_type, num_discretize[0], num_discretize[1], now.strftime('%Y%m%d%H%M'))
    save_name_s = "st_{}_state({}-{})_{}".format(model_type, num_discretize[0], num_discretize[1], now.strftime('%Y%m%d%H%M'))
    save_name_n = "num_visits_state({}-{})_{}".format( num_discretize[0], num_discretize[1], now.strftime('%Y%m%d%H%M'))
    os.makedirs(output_dir, exist_ok=True)
    square_plot(ht_all_norm, save_name=save_name_h, lims=[-1.0,1.0])
    # square_plot(ht_all_norm, save_name=save_name_h, lims=[ht_all_norm.min(),ht_all_norm.max()])
    square_plot(st_all_norm, save_name=save_name_s, lims=[st_all_norm.min(),st_all_norm.max()])
    square_plot(num_visits, save_name=save_name_n, lims=[num_visits.min(),num_visits.max()])
    # square_plot(ht_all_norm, save_name=save_name_h)
    # square_plot(st_all_norm, save_name=save_name_s)
    # square_plot(num_visits, save_name=save_name_n)
else:
    for direction_state in range(num_discretize[2]):
        save_name_h = "ht_{}_state({}-{}-{}_{})_{}".format(model_type, num_discretize[0], num_discretize[1], direction_state+1,num_discretize[2], now.strftime('%Y%m%d%H%M'))
        save_name_s = "st_{}_state({}-{}-{}_{})_{}".format(model_type, num_discretize[0], num_discretize[1], direction_state+1,num_discretize[2], now.strftime('%Y%m%d%H%M'))
        save_name_n = "num_visits_state({}-{}-{}_{})_{}".format(num_discretize[0], num_discretize[1], direction_state+1,num_discretize[2], now.strftime('%Y%m%d%H%M'))
        # os.makedirs(output_dir, exist_ok=True)
        square_plot(ht_all_norm, save_name=save_name_h, lims=[-1.0,1.0], direction_state=direction_state)
        # square_plot(ht_all_norm, save_name=save_name_h, lims=[ht_all_norm.min(),ht_all_norm.max()], direction_state=direction_state)
        square_plot(st_all_norm, save_name=save_name_s, lims=[st_all_norm.min(),st_all_norm.max()], direction_state=direction_state)
        square_plot(num_visits, save_name=save_name_n, lims=[num_visits.min(),num_visits.max()], direction_state=direction_state)
    
    
