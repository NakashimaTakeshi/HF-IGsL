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

# 勾配を計算しない
torch.set_grad_enabled(False)

#パラーメーター設定
path_name = "HF-PGM_model2-seed_0/2023-01-26/run_2"
model_idx = 2
cfg_device = "cuda:1"
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

# use_validation_data = False

#load data
if data=="train":
    D = get_dataset_loader(cfg, cwd, device, cfg.train.train_data_path)
elif data=="validation":
    D = get_dataset_loader(cfg, cwd, device, cfg.train.validation_data_path)
else:
    raise NotImplementedError

# D_test = get_dataset_loader(cfg, cwd, device, cfg.train.test_data_path)


# # Reconstruction
epi_idx = 0
crop_idx = 0
observations, actions, rewards, nonterminals = get_episode_data(D, epi_idx = epi_idx, crop_idx = crop_idx)

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

ht = [states_np[key]["beliefs"] for key in states_np.keys()]
ht = np.vstack(ht)
pca_ht = get_pca_model(ht, 3)
pca_ht_2d = get_pca_model(ht, 2)
st_q = [states_np[key]["posterior_means"] for key in states_np.keys()]
st_q = np.vstack(st_q)
pca_st_q = get_pca_model(st_q, 3)
pca_st_q_2d = get_pca_model(st_q, 2)

#再構成
observations_target = model._clip_obs(observations, idx_start=1)
#estimate_state   subset_index(0:空集合、1:image、2:Pose、3:両方)
state = model.estimate_state(observations_target, actions[:-1], rewards, nonterminals[:-1], subset_index = 1)
recon = model.observation_model(h_t=state["beliefs"], s_t=state["posterior_means"])
recon["state"]={"beliefs":state["beliefs"],"posterior_means":state["posterior_means"]}

pose_predict_loc = recon['Pose']['loc'].detach().cpu().numpy()
pose_predict_scale = recon['Pose']['scale'].detach().cpu().numpy()

print(pose_predict_loc[0])

#位置尤度計算
# pose_predict_loc = []
# pose_predict_scale = []

# for t in range(len(actions)-1):
#     pose_predict_loc.append(tensor2np(model.pose_poredict_model(state["beliefs"][t])["loc"]))
#     pose_predict_scale.append(tensor2np(model.pose_poredict_model(state["beliefs"][t])["scale"]))

# pose_predict_loc = np.stack(pose_predict_loc)


#部屋ラベル分け
room_label = []
for t in range(len(actions)-1):
    room_label.append(room_clustering(observations_target['Pose'].detach().cpu().numpy()[t, 0]))
room_label = np.stack(room_label)
print(room_label)

#潜在空間表現(再構成)(PCA)
ht_recon = recon["state"]["beliefs"]
shape = ht_recon.shape[-1]
feat = tensor2np(ht_recon).reshape(-1, shape)
feat_pca = pca_ht.transform(feat)
hx_recon, hy_recon, hz_recon = get_xyz(feat_pca)

feat_pca = pca_ht_2d.transform(feat)
hx_recon_2d, hy_recon_2d = get_xy(feat_pca)

st_q_recon = recon["state"]["posterior_means"]
shape = st_q_recon.shape[-1]
feat = tensor2np(st_q_recon).reshape(-1, shape)
feat_pca = pca_st_q.transform(feat)
sx_recon, sy_recon, sz_recon = get_xyz(feat_pca)

feat_pca = pca_st_q_2d.transform(feat)
sx_recon_2d, sy_recon_2d = get_xy(feat_pca)

#潜在空間表現(再構成)(T-SNE)
tsne = TSNE(n_components = 2)


ht_tsne = tsne.fit_transform(ht_recon.detach().cpu().numpy()[:,0])
st_q_tsne = tsne.fit_transform(st_q_recon.detach().cpu().numpy()[:,0])

fig_tsne = plt.figure(figsize=(10,5))
ax1 = fig_tsne.add_subplot(121)
ax2 = fig_tsne.add_subplot(122)
# ax1.set_aspect('equal')
# ax2.set_aspect('equal')
cmap = plt.get_cmap("tab10") 
# for i in tqdm(range(len(ht_tsne))):
#     ax1.scatter(ht_tsne[i][0], ht_tsne[i][1], color = cmap(room_label[i]), s = 2)
#     ax2.scatter(hx_recon_2d[i],hy_recon_2d[i], color = cmap(room_label[i]), s = 2)
# for i in tqdm(range(len(ht_tsne)-2)):
#     ax1.plot(ht_tsne[i:i+2, 0], ht_tsne[i:i+2, 1], color = cmap(room_label[i]))
#     ax2.plot(hx_recon_2d[i:i+2],hy_recon_2d[i:i+2], color = cmap(room_label[i]))
# fig_tsne.savefig("tsne_ep{}.png".format(epi_idx))

np.save("ht_tsne_model2.npy",ht_tsne, allow_pickle=True)
np.save("ht_pca_model2.npy",[hx_recon_2d, hy_recon_2d], allow_pickle=True)
# np.save("ht_pca_model2.npy",[hx_recon, hy_recon, hz_recon], allow_pickle=True)
np.save("room_model2.npy", room_label, allow_pickle=True)

#イマジネーション
t_imag_start = 30

h_t_img = [state["beliefs"][t_imag_start]]
s_t_img = [state["posterior_means"][t_imag_start]]

t_max = len(actions)
horizon_imagination = t_max-t_imag_start
for t in range(horizon_imagination):
    belief, _, prior_mean, _ = model.transition_model(s_t_img[t], actions[t_imag_start+t].unsqueeze(dim=0), h_t_img[t], det=True)
    h_t_img.append(belief.squeeze(dim=0))
    # st.append(prior_state.squeeze(dim=0))
    s_t_img.append(prior_mean.squeeze(dim=0))

h_t_img = torch.stack(h_t_img)
s_t_img = torch.stack(s_t_img)

recon_imag = model.observation_model(h_t=h_t_img, s_t=s_t_img)
recon_imag["state"]={"beliefs":h_t_img,"posterior_means":s_t_img}

#潜在空間表現(imgnation)(PCA)
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
# ax2.set_aspect('equal')
w_graph = 4
h_graph = 2

fig = plt.figure(figsize=(w_graph*5,h_graph*5))
ax1 = fig.add_subplot(h_graph, w_graph, 1)
ax2 = fig.add_subplot(h_graph, w_graph, 2)
ax3 = fig.add_subplot(h_graph, w_graph, 3)
ax4 = fig.add_subplot(h_graph, w_graph, 4)
ax5 = fig.add_subplot(h_graph, w_graph, 5)
ax6 = fig.add_subplot(h_graph, w_graph, 6)
ax7 = fig.add_subplot(h_graph, w_graph, 7, projection="3d")
ax8 = fig.add_subplot(h_graph, w_graph, 8, projection="3d")


image_name_horizon="image_hsr_256"
n_frame = len(recon[image_name_horizon]["loc"])
# n_frame = len(recon["loc"])

dt = 1
size = (256,256)
Pose_list = np.empty((0,4))
long=20
img_map=plt.imread("map_reshape_2.bmp")
color_list = ["orange","pink","blue","red","yellow","green"]
plot_array_recon_ht = [[],[],[],[],[],[]]
plot_array_recon_st = [[],[],[],[],[],[]]
print(len(plot_array_recon_ht))

def plot_rcon(t):
    global Pose_list
    plt.cla()
    ax1.cla()
    ax2.cla()
    ax3.cla()
    ax4.cla()
    ax5.cla()
    ax6.cla()
    ax7.cla()
    ax8.cla()

    ax1.axis("off")
    ax2.axis("off")
    ax3.axis("on") 
    ax4.axis("on") 
    ax5.axis("off")
    ax6.axis("off")
    ax7.axis("on") 
    ax8.axis("on") 

    fig.suptitle("time t={}s (Speed: 3x)".format(round(t)))

    im = observations[image_name_horizon][t, 0][[2, 1, 0]].detach().cpu().numpy()
    # im = observations[t, 0][[2,1,0]].detach().cpu().numpy()
    im = reverse_image_observation(im)
    im = cv2.resize(im, size, interpolation=cv2.INTER_LINEAR)
    ax1.imshow(im)
    ax1.set_title("Observation")
    
    im = recon[image_name_horizon]["loc"][t, 0][[2, 1, 0]].detach().cpu().numpy()
    # im = recon["loc"][t, 0][[2,1,0]].detach().cpu().numpy()
    im = reverse_image_observation(im)
    im = cv2.resize(im, size, interpolation=cv2.INTER_LINEAR)
    ax2.imshow(im)
    ax2.set_title("Reconstruction")
    # pint(t)
    
    # ax3.plot(hx_recon[0:t], hy_recon[0:t], hz_recon[0:t], alpha=0.4)
    # ax3.scatter(hx_recon[t],hy_recon[t],hz_recon[t],label="h")
    
    plot_array_recon_ht[room_label[t]].append(np.array([hx_recon_2d[t], hy_recon_2d[t]]))

    for j in range(6):
        if len(plot_array_recon_ht[j]) == 0:
            continue
        else:
            plot_nparray_recon_ht = np.array(plot_array_recon_ht[j])
            ax3.scatter(plot_nparray_recon_ht[:, 0], plot_nparray_recon_ht[:, 1], color = color_list[j], label="h")

    ax3.plot(hx_recon_2d[0:t], hy_recon_2d[0:t], alpha=0.4)
    ax3.scatter(hx_recon_2d[t], hy_recon_2d[t], label="h")
    ax3.set_xlim(-10, 10)
    ax3.set_ylim(-10, 10)
    # ax3.set_zlim(-10,10)
    ax3.set_title("Reconstruction h(t)")
    #ax3.legend()


    plot_array_recon_st[room_label[t]].append(np.array([sx_recon_2d[t], sy_recon_2d[t]]))

    for j in range(6):
        if len(plot_array_recon_st[j]) == 0:
            continue
        else:
            plot_nparray_recon_st = np.array(plot_array_recon_st[j])
            ax4.scatter(plot_nparray_recon_st[:, 0], plot_nparray_recon_st[:, 1], color = color_list[j], label="h")
    # ax4.plot(sx_recon[0:t], sy_recon[0:t], sz_recon[0:t], alpha=0.4)
    # ax4.scatter(sx_recon[t],sy_recon[t],sz_recon[t],label="s")
    # ax4.plot(sx_recon_2d[0:t], sy_recon_2d[0:t], alpha = 0.4)
    # ax4.scatter(sx_recon_2d[t],sy_recon_2d[t], label = "s")
    ax4.set_xlim(-30, 30)
    ax4.set_ylim(-30, 30)
    #ax4.set_zlim(-30,30)
    ax4.set_title("Reconstruction s(t)")
    #ax4.legend()

    t_obs = t - t_imag_start
    
    if t<t_imag_start:
        im = recon[image_name_horizon]["loc"][t, 0][[2, 1, 0]].detach().cpu().numpy()
        # im = recon["loc"][t, 0][[2,1,0]].detach().cpu().numpy()
        im = reverse_image_observation(im)
        im = cv2.resize(im, size, interpolation = cv2.INTER_LINEAR)
        ax6.imshow(im)
        ax6.set_title("Imagination [Input Data]")
        # pint(t)
        
        ax7.plot(hx_recon[0:t], hy_recon[0:t], hz_recon[0:t], alpha = 0.4)
        ax7.scatter(hx_recon[t], hy_recon[t], hz_recon[t], label = "h")
        ax7.set_xlim(-10, 10)
        ax7.set_ylim(-10, 10)
        ax7.set_zlim(-10, 10)
        ax7.set_title("Imagination h(t) [Input Data]")
        #ax3.legend()

        ax8.plot(sx_recon[0:t], sy_recon[0:t], sz_recon[0:t], alpha=0.4)
        ax8.scatter(sx_recon[t], sy_recon[t], sz_recon[t], label="s")
        ax8.set_xlim(-30, 30)
        ax8.set_ylim(-30, 30)
        ax8.set_zlim(-30, 30)
        ax8.set_title("Imagination s(t) [Input Data]")
    else:
        im = recon_imag[image_name_horizon]["loc"][t_obs, 0][[2,1,0]].detach().cpu().numpy()
        # im = recon_imag["loc"][t_obs, 0][[2,1,0]].detach().cpu().numpy()
        im = reverse_image_observation(im)
        im = cv2.resize(im, size, interpolation=cv2.INTER_LINEAR)
        ax6.imshow(im)
        ax6.set_title("Imagination")
        # pint(t)
        
        ax7.plot(hx_img[t_imag_start:t_obs], hy_img[t_imag_start:t_obs], hz_img[t_imag_start:t_obs], alpha=0.4)
        ax7.scatter(hx_img[t_obs],hy_img[t_obs],hz_img[t_obs],label="h")
        ax7.set_xlim(-10,10)
        ax7.set_ylim(-10,10)
        ax7.set_zlim(-10,10)
        ax7.set_title("Imagination h(t)")

        ax8.plot(sx_img[t_imag_start:t_obs], sy_img[t_imag_start:t_obs], sz_img[t_imag_start:t_obs], alpha=0.4)
        ax8.scatter(sx_img[t_obs], sy_img[t_obs], sz_img[t_obs], label="s")
        ax8.set_xlim(-30,30)
        ax8.set_ylim(-30,30)
        ax8.set_zlim(-30,30)
        ax8.set_title("Imagination s(t)")
    
    
    Pose_list=np.append(Pose_list,display_pose(observations["Pose"][t, 0].detach().cpu().numpy()),axis=0)

    ax5.imshow(img_map)
    ax5.scatter(Pose_list[t,0],Pose_list[t,1],s=5,c="red")
    
    x_end=long*np.cos((Pose_list[t,3]/180)*math.pi)+Pose_list[t,0]
    y_end=-1*long*np.sin((Pose_list[t,3]/180)*math.pi)+Pose_list[t,1]
    ax5.annotate('', xy=[x_end,y_end], xytext=[Pose_list[t,0],Pose_list[t,1]],
            arrowprops=dict(shrink=0, width=1, headwidth=3, 
                            headlength=5, connectionstyle='arc3',
                            facecolor='red', edgecolor='red')
            )
    ax5.plot(Pose_list[0:t,0],Pose_list[0:t,1])

    #位置尤度の出力
    ax5.scatter(pose_predict_loc[t,0,0]*20+190,pose_predict_loc[t,0,1]*(-20)+115, s = 8, c="green")

    E = pat.Ellipse(xy=(pose_predict_loc[t,0,0]*20+190, pose_predict_loc[t,0,1]*(-20)+115), width=pose_predict_scale[t,0,0]*20*1.6*2, height=pose_predict_scale[t,0,1]*20*1.6*2, color="lime", alpha=0.2)

    ax5.add_patch(E)
    ax5.set_title("Action (Pose)")

    # plt.savefig(save_file_name)


# 4. アニメーション化
anim = FuncAnimation(fig, plot_rcon, tqdm(np.arange(n_frame)), interval= dt*333)

os.makedirs(folder_name, exist_ok=True)
if data=="train":
    save_file_name = f"{folder_name}/train_ep{epi_idx}_predict_pose.mp4"
else:
    now = datetime.datetime.now()
    save_file_name = "{}/validation_ep{}_predict_pose_".format(folder_name, epi_idx) + now.strftime('%Y_%m%d_%H%M%S') + '.mp4'
anim.save(save_file_name, writer='ffmpeg')

print(save_file_name)
print("fin")

def cluster_room_plot(colllor_room, x, y, ax, start = 0):
    plot_array = [[]*6]
    for i in len(x + start):
        plot_array[colllor_room[i]].aapend = [x[i], y[i]]
    

    
    return plot_array

