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
from IPython import display
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ArtistAnimation

import numpy as np
from tqdm import tqdm


module_path = os.path.join(Path().resolve(), '../../../../..')
sys.path.append(module_path)

#pythonpathの追加
os.environ['PYTHONPATH'] = module_path

# 勾配を計算しない
torch.set_grad_enabled(False)


model_folder = "results/test2-seed_0/2022-12-07/run_0"
folder_name = "figs_out_dir/test"
model_idx = 3
cfg_device = "cuda:1"
#cfg_device = "cpu"
cwd = "."

from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from omegaconf import OmegaConf
with initialize(config_path=model_folder):
    cfg = compose(config_name="hydra_config")



cfg.main.device = cfg_device
# cfg.train.n_episode = 100
print(' ' * 26 + 'Options')
for k, v in cfg.items():
    print(' ' * 26 + k + ': ' + str(v))

device = torch.device(cfg.main.device)



# # Load Model, Data and States

import glob
model_paths = glob.glob(os.path.join(model_folder, '*.pth'))
print("model_pathes: ")
model_paths





from algos.MRSSM.MRSSM.algo import build_RSSM

model = build_RSSM(cfg, device)
model_path = model_paths[model_idx]
model.load_model(model_path)
model.eval()


use_validation_data = False


from algos.MRSSM.MRSSM.train import get_dataset_loader

#load data

# if use_validation_data:
#     D = get_dataset_loader(cfg, cwd, device, cfg.train.validation_data_path)
# else:
#     D = get_dataset_loader(cfg, cwd, device, cfg.train.train_data_path)
print(cfg.rssm.condition_names)
cfg.env.action_name="Pose"

D_val = get_dataset_loader(cfg, cwd, device, cfg.train.validation_data_path)
D_train = get_dataset_loader(cfg, cwd, device, cfg.train.train_data_path)
# D_test = get_dataset_loader(cfg, cwd, device, cfg.train.test_data_path)


# # Reconstruction
from utils.evaluation.estimate_states import get_episode_data

epi_idx = 0
crop_idx = 0
observations, actions, rewards, nonterminals = get_episode_data(D_val, epi_idx=epi_idx, crop_idx=crop_idx)


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




from utils.evaluation.visualize_utils import get_pca_model, tensor2np, get_xyz


# #### <span style="color: red; ">please run estimate_state.py for estimate states of train dataset</span>


# load states
state_path = model_path.replace("models", "states_models").replace(".pth", ".npy")
print("state_path:", state_path)

states_np = np.load(state_path, allow_pickle=True).item()
print("-- dataset --")
for key in states_np.keys():
    print(key)

print("-- key of states --")
print(states_np[key].keys())



ht = [states_np[key]["beliefs"] for key in states_np.keys()]
ht = np.vstack(ht)
pca_ht = get_pca_model(ht)



st_q = [states_np[key]["posterior_means"] for key in states_np.keys()]
st_q = np.vstack(st_q)
pca_st_q = get_pca_model(st_q)




fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection="3d")

for key in states_np.keys():
    ht = states_np[key]["beliefs"]
    shape = ht.shape[-1]
    feat = tensor2np(ht).reshape(-1, shape)
    feat_pca = pca_ht.transform(feat)
    x, y, z = get_xyz(feat_pca)
    ax.plot(x, y, z, alpha=0.2)
    # ax.scatter(x, y, z, alpha=0.2)
plt.show()



fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection="3d")

for key in states_np.keys():
    st_q = states_np[key]["posterior_means"]
    shape = st_q.shape[-1]
    feat = tensor2np(st_q).reshape(-1, shape)
    feat_pca = pca_st_q.transform(feat)
    x, y, z = get_xyz(feat_pca)
    ax.plot(x, y, z, alpha=0.2)
    # ax.scatter(x, y, z, alpha=0.2)
plt.show()

observations_target = model._clip_obs(observations, idx_start=1)
state = model.estimate_state(observations_target, actions[:-1], rewards, nonterminals[:-1])


# recon = model.rssm.observation_model(h_t=state["beliefs"], s_t=state["posterior_states"])
recon = model.observation_model(h_t=state["beliefs"], s_t=state["posterior_means"])

recon["state"]={"beliefs":state["beliefs"],"posterior_means":state["posterior_means"]}





# get_ipython().system('pip install opencv-python')
import cv2
from utils.evaluation.visualize_utils import reverse_image_observation
from IPython import display
import mpl_toolkits

h_graph = 2
w_graph = 2
fig = plt.figure(figsize=(w_graph*5,h_graph*5))
ax1 = fig.add_subplot(h_graph, w_graph, 1)
ax2 = fig.add_subplot(h_graph, w_graph, 2)
ax3 = fig.add_subplot(h_graph, w_graph, 3, projection="3d")
ax4 = fig.add_subplot(h_graph, w_graph, 4, projection="3d")

image_name_horizon="image_hsr_256"
n_frame = len(recon[image_name_horizon]["loc"])



colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]

dt = 1
size = (256,256)
# artists = []
# # for t in range(0, n_frame, dt):
#     im = observations[image_name_horizon][t, 0][[2,1,0]].detach().cpu().numpy()
#     im = reverse_image_observation(im)
#     im = cv2.resize(im, size, interpolation=cv2.INTER_LINEAR)
#     im1 = ax1.imshow(im)
#     ax1.set_title("Observation")
    
#     im = recon[image_name_horizon]["loc"][t, 0][[2,1,0]].detach().cpu().numpy()
#     im = reverse_image_observation(im)
#     im = cv2.resize(im, size, interpolation=cv2.INTER_LINEAR)
#     im2 = ax2.imshow(im)
#     ax2.set_title("Reconstruction")
#     # print(t)
    
#     ax3.plot(x[0:t], y[0:t], z[0:t], alpha=0.2)
#     ax3.scatter(x[t],y[t],z[t])

    
#     #artists.append([im1,im2,im3,im4])
#     artists.append([im1,im2])

ht = recon["state"]["beliefs"]
shape = ht.shape[-1]
feat = tensor2np(ht).reshape(-1, shape)
feat_pca = pca_ht.transform(feat)
hx, hy, hz = get_xyz(feat_pca)


st_q = recon["state"]["posterior_means"]
shape = st_q.shape[-1]
feat = tensor2np(st_q).reshape(-1, shape)
feat_pca = pca_st_q.transform(feat)
sx, sy, sz = get_xyz(feat_pca)

print("ht[0]:{},ht[-1]:{}".format(ht[0],ht[-1]))
print("st[0]:{},st[-1]:{}".format(st_q[0],st_q[-1]))
# def plot_pca(t,state,ax):
#     feat_pca=state
#     x, y, z = get_xyz(feat_pca)
#     ax.plot(x[0:t], y[0:t], z[0:t])
#     ax.scatter(x[t],y[t],z[t])


def plot_rcon(t):
    plt.cla()
    ax1.cla()
    ax2.cla()
    ax3.cla()
    ax4.cla()

    ax1.axis("off")
    ax2.axis("off")
    ax3.axis("on") 
    ax4.axis("on") 

    fig.suptitle("time t={}s".format(round(t/10,1)))

    im = observations[image_name_horizon][t, 0][[2,1,0]].detach().cpu().numpy()
    im = reverse_image_observation(im)
    im = cv2.resize(im, size, interpolation=cv2.INTER_LINEAR)
    ax1.imshow(im)
    ax1.set_title("Observation")
    
    im = recon[image_name_horizon]["loc"][t, 0][[2,1,0]].detach().cpu().numpy()
    im = reverse_image_observation(im)
    im = cv2.resize(im, size, interpolation=cv2.INTER_LINEAR)
    ax2.imshow(im)
    ax2.set_title("Reconstruction")
    # pint(t)
    
    ax3.plot(hx[0:t], hy[0:t], hz[0:t], alpha=0.4)
    ax3.scatter(hx[t],hy[t],hz[t],label="h")
    ax3.set_xlim(10,-10)
    ax3.set_ylim(10,-10)
    ax3.set_zlim(10,-10)
    ax3.set_title("h(t)")
    #ax3.legend()

    ax4.plot(sx[0:t], sy[0:t], sz[0:t], alpha=0.4)
    ax4.scatter(sx[t],sy[t],sz[t],label="s")
    ax4.set_xlim(5,-5)
    ax4.set_ylim(5,-5)
    ax4.set_zlim(5,-5)
    ax4.set_title("s(t)")
    #ax4.legend()


    # plt.savefig(save_file_name)


# 4. アニメーション化
# anim = ArtistAnimation(fig, tqdm(artists), interval=100*dt)
anim = FuncAnimation(fig, plot_rcon, frames=n_frame, interval= dt*100)

# if use_validation_data:
#     folder_name = "figs_out_dir/20221127run1/validation"
# else:
#     folder_name = "figs_out_dir/20221127run1/train"

os.makedirs(folder_name, exist_ok=True)
save_file_name = "{}/reconstruction_ep{}.mp4".format(folder_name, epi_idx)
anim.save(save_file_name, writer='ffmpeg')
# for i in range(0,n_frame,1):
#     save_file_name = "{}/reconstruction_ep{}_{}.png".format(folder_name, epi_idx,i)
#     plot_rcon(i)

plt.show()

print("fin")

print("state_keys:",state.keys())
# expert_means = state["expert_means"]
# expert_std_devs = state["expert_std_devs"]

prior_means = state["prior_means"][:,0].detach().cpu().numpy()
prior_std_devs = state["prior_std_devs"][:,0].detach().cpu().numpy()

posterior_means = state["posterior_means"][:,0].detach().cpu().numpy()
posterior_std_devs = state["posterior_std_devs"][:,0].detach().cpu().numpy()

# expert_means = dict()
# expert_std_devs = dict()
# for name in state["expert_means"].keys():
#     expert_means[name] = state["expert_means"][name][:,0].detach().cpu().numpy()
#     expert_std_devs[name] = state["expert_std_devs"][name][:,0].detach().cpu().numpy()



colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
# expert_colors = dict()
# keys = list(expert_means.keys())
# for i in range(len(keys)):
#     expert_colors[keys[i]] = colors[i+2]

# expert_means.keys()

# expert_means["prior_expert"].shape[1]


from utils.dist import calc_subset_states

# subset_means, subset_std_devs = calc_subset_states(state["expert_means"], state["expert_std_devs"])

# len(subset_means)

t_imag_start = 100

h_t = [state["beliefs"][t_imag_start]]
# s_t = [state["posterior_states"][t_imag_start]]
s_t = [state["posterior_means"][t_imag_start]]

t_max = len(actions)
horizon_imagination = t_max-t_imag_start
for t in range(horizon_imagination):
    belief, _, prior_mean, _ = model.transition_model(s_t[t], actions[t_imag_start+t].unsqueeze(dim=0), h_t[t], det=True)
    h_t.append(belief.squeeze(dim=0))
    # st.append(prior_state.squeeze(dim=0))
    s_t.append(prior_mean.squeeze(dim=0))

print("actions=")
print(actions)



h_t = torch.stack(h_t)
s_t = torch.stack(s_t)



recon_imag = model.observation_model(h_t=h_t, s_t=s_t)
recon_imag["state"]={"beliefs":h_t,"posterior_means":s_t}
# recon_imag["image_vertical_high_resolution"]["loc"].shape




import cv2
from utils.evaluation.visualize_utils import reverse_image_observation
from IPython import display
import mpl_toolkits

h_graph = 2
w_graph = 2
fig = plt.figure(figsize=(w_graph*5,h_graph*5))
ax1 = fig.add_subplot(h_graph, w_graph, 1)
ax2 = fig.add_subplot(h_graph, w_graph, 2)
ax3 = fig.add_subplot(h_graph, w_graph, 3,projection="3d")
ax4 = fig.add_subplot(h_graph, w_graph, 4,projection="3d")

n_frame = horizon_imagination



colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]

dt = 1
size = (256,256)
# artists = []
# for t in range(0, n_frame, dt):
#     t_obs = t_imag_start+t
    
#     if t_obs >= len(observations[image_name_horizon]):
#         t_obs = len(observations[image_name_horizon])-1
    
#     im = observations[image_name_horizon][t_obs, 0][[2,1,0]].detach().cpu().numpy()
#     im = reverse_image_observation(im)
#     im = cv2.resize(im, size, interpolation=cv2.INTER_LINEAR)
#     im1 = ax1.imshow(im)
#     ax1.set_title("Observation")
    
#     im = recon_imag[image_name_horizon]["loc"][t, 0][[2,1,0]].detach().cpu().numpy()
#     im = reverse_image_observation(im)
#     im = cv2.resize(im, size, interpolation=cv2.INTER_LINEAR)
#     im2 = ax2.imshow(im)
#     ax2.set_title("Imagination")
    
#     im = observations["sound"][t_obs, 0].detach().cpu().numpy()
#     # im = image_postprocess(im)
#     im = cv2.resize(im, size, interpolation=cv2.INTER_LINEAR)
#     im3 = ax3.imshow(im)
#     ax3.set_title("Observation")
    
#     im = recon_imag["sound"]["loc"][t, 0].detach().cpu().numpy()
#     # im = image_postprocess(im)
#     im = cv2.resize(im, size, interpolation=cv2.INTER_LINEAR)
#     im4 = ax4.imshow(im)
#     ax4.set_title("Imagination")
    
    
# #     artists.append([im1,im2,im3,im4])
#     artists.append([im1,im2])

# # 4. アニメーション化
# anim = ArtistAnimation(fig, tqdm(artists), interval=100*dt)

ht = recon_imag["state"]["beliefs"]
shape = ht.shape[-1]
feat = tensor2np(ht).reshape(-1, shape)
feat_pca = pca_ht.transform(feat)
hx, hy, hz = get_xyz(feat_pca)
print("hx:{},hy:{},hz:{}".format(hx,hy,hz))

st_q = recon_imag["state"]["posterior_means"]
shape = st_q.shape[-1]
feat = tensor2np(st_q).reshape(-1, shape)
feat_pca = pca_st_q.transform(feat)
sx, sy, sz = get_xyz(feat_pca)

def plot_img(t):
    plt.cla()
    ax1.cla()
    ax2.cla()
    ax3.cla()
    ax4.cla()

    ax1.axis("off")
    ax2.axis("off")
    ax3.axis("on") 
    ax4.axis("on") 

    t_obs = t_imag_start+t
    if t_obs >= len(observations[image_name_horizon]):
        t_obs = len(observations[image_name_horizon])-1

    fig.suptitle("time t={}s".format(round(t_obs/10,1)))
    im = observations[image_name_horizon][t_obs, 0][[2,1,0]].detach().cpu().numpy()
    im = reverse_image_observation(im)
    im = cv2.resize(im, size, interpolation=cv2.INTER_LINEAR)
    ax1.imshow(im)
    ax1.set_title("Observation")
    
    im = recon_imag[image_name_horizon]["loc"][t_obs, 0][[2,1,0]].detach().cpu().numpy()
    im = reverse_image_observation(im)
    im = cv2.resize(im, size, interpolation=cv2.INTER_LINEAR)
    ax2.imshow(im)
    ax2.set_title("Imagination")
    # pint(t)
    
    ax3.plot(hx[t_imag_start:t_obs], hy[t_imag_start:t_obs], hz[t_imag_start:t_obs], alpha=0.4)
    ax3.scatter(hx[t_obs],hy[t_obs],hz[t_obs],label="h")
    ax3.set_xlim(10,-10)
    ax3.set_ylim(10,-10)
    ax3.set_zlim(10,-10)
    ax3.set_title("h(t)")

    ax4.plot(sx[t_imag_start:t_obs], sy[t_imag_start:t_obs], sz[t_imag_start:t_obs], alpha=0.4)
    ax4.scatter(sx[t_obs],sy[t_obs],sz[t_obs],label="s")
    ax4.set_xlim(5,-5)
    ax4.set_ylim(5,-5)
    ax4.set_zlim(5,-5)
    ax4.set_title("s(t)")

    # plt.savefig(save_file_name)


anim = FuncAnimation(fig, plot_img, frames=n_frame, interval= dt*100)
# if use_validation_data:
#     folder_name = "figs_out_dir/20221127run1/validation"
# else:
#     folder_name = "figs_out_dir/20221127run1/train"

os.makedirs(folder_name, exist_ok=True)
# for t in range(0,n_frame,1):
#     save_file_name = "{}/imagination_ep{}_{}.png".format(folder_name, epi_idx,t)
#     plot_img(t)
save_file_name = "{}/imagination_ep{}.mp4".format(folder_name, epi_idx)
anim.save(save_file_name, writer='ffmpeg')
plt.show()

print("fin")





