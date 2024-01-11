#!/usr/bin/env python
# coding: utf-8
import sys
import os
import math
import datetime

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
from sklearn.manifold import TSNE
import matplotlib.patches as pat
from mpl_toolkits.mplot3d import Axes3D

from PIL import Image

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
from scipy.spatial.transform import Rotation as R


def transform_pose(pose):
    """
    Transforms a pose for visualization.

    Args:
        pose (list): The pose to be transformed. It should be a list of either 4 or 7 elements.
    """
    pose_array = np.zeros_like(pose, dtype=float)
    pose_array[:,0:2] = pose[:,0:2]

    if pose.shape[-1] == 4:
        pose_array[-1,3] = math.degrees(math.atan2(pose[-1,3], pose[-1,2]))
    else:
        raise NotImplementedError

    arrow_length = 1
    arrow_end_x = pose_array[-1,0] + arrow_length * math.cos(math.radians(pose_array[-1,3]))
    arrow_end_y = pose_array[-1,1] + arrow_length * math.sin(math.radians(pose_array[-1,3]))

    return pose_array,[arrow_end_x,arrow_end_y]

def room_clustering(position):
    x = position[0]
    y = position[1]

    id_room = 0
    
    if x <= -1.0:
        if y < 2.0:
            id_room = 1
        else:
            id_room = 2
    elif -1.0 < x and x <=1.0:
        if -1.5 < y and y <= 0.6:
            id_room = 3
        elif 0.6 < y:
            id_room = 4
    elif 1.0 < x and x <= 2.0:
        if 3.0 < y:
            id_room = 5
    elif 2.0 < x and x <= 3.0:
        if y <= -2.0:
            id_room = 6
    elif 3.0 < x:
        if y <= -2.0:
            id_room = 6
        elif -2.0 < y and y <= 1.6:
            id_room = 7
        elif 1.0 < y:
            id_room = 8

    # aws small house 
    # if x <= -2.5:
    #     if y > -1.5:
    #         room = 0
    #     else:
    #         room = 1
    # elif x > -2.5 and x <=4.75:
    #     if y > 1.3:
    #         room = 2
    #     else:
    #         room = 3
    # else:
    #     if y > -1.3:
    #         room = 4
    #     else:
    #         room = 5

    return id_room

def convert_elements(arr, table):
    return np.array([table.get(item, item) for item in arr])

def load_map(map_path):
    import yaml

    yaml_path = map_path.replace(".pgm", ".yaml")
    with open(yaml_path, 'r') as file:
        map_info = yaml.safe_load(file)
    map_image=Image.open(map_path)
    return map_image, map_info

def get_ellipse(scale_x, scale_y):
    import scipy.stats as ss
    sigma_matrix = [[scale_x, 0], [0, scale_y]]

    el_prob  = 0.80
    el_c     = np.sqrt(ss.chi2.ppf(el_prob, 2))
    
    lmda, vec            = np.linalg.eig(sigma_matrix)
    el_width,el_height   = 2 * el_c * np.sqrt(lmda)
    el_angle             = np.rad2deg(np.arctan2(vec[1,0],vec[0,0]))
    # el                   = Ellipse(xy=mu_k[k],width=el_width,height=el_height,angle=el_angle,color=colorlist[k],alpha=0.3)
    return el_width, el_height, el_angle

def main():

    # 勾配を計算しない
    torch.set_grad_enabled(False)

    # Load Config
    train_log_dir = "HF-PGM_model2-seed_0/2023-11-26/run_1"
    # train_log_dir = "HF-PGM_model1-seed_0/2023-11-26/run_2"
    mun_iteration = "_3000.pth"
    # data="train"
    data="validation"
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

    # load states. Note, please run estimate_state.py for estimate states of train dataset
    state_path = model_path.replace("models", "states_models").replace(".pth", ".npy")
    print("Loading states from :", state_path)
    states_all = np.load(state_path, allow_pickle=True).item() 
    print("-- dataset --")
    for key in states_all.keys():
        print(key)

    print("-- key of states --")
    print(states_all[key].keys())

    ht = [states_all[key]["beliefs"] for key in states_all.keys()]
    ht = np.vstack(ht)
    pca_ht_3d = get_pca_model(ht, 3)
    pca_ht_2d = get_pca_model(ht, 2)

    # We use posterior mean for visualization
    st_q = [states_all[key]["posterior_means"] for key in states_all.keys()]
    st_q = np.vstack(st_q)
    pca_st_q_3d = get_pca_model(st_q, 3)
    pca_st_q_2d = get_pca_model(st_q, 2)

    #load target data
    if data=="train":
        D = get_dataset_loader(cfg, cwd, device, cfg.train.train_data_path)
    elif data=="validation":
        D = get_dataset_loader(cfg, cwd, device, cfg.train.validation_data_path)
    else:
        raise NotImplementedError

    epi_idx = 0
    observations, actions, rewards, nonterminals = get_episode_data(D, epi_idx = epi_idx)

    # Reconstruction
    observations_target = model._clip_obs(observations, idx_start=1)
    # state = model.estimate_state(observations_target, actions[:-1], rewards, nonterminals[:-1], det=True)
    state = model.estimate_state(observations_target, actions[:-1], rewards, nonterminals[:-1], subset_index = 1,det=True)
    reconstructions = model.observation_model(h_t=state["beliefs"], s_t=state["posterior_means"])

    if "Pose" in reconstructions.keys():
        predicted_pose_loc = reconstructions['Pose']['loc'].detach().cpu().numpy()
        predicted_pose_scale = reconstructions['Pose']['scale'].detach().cpu().numpy()
        input_pose = observations_target['Pose'].detach().cpu().numpy()

    else :
        predicted_pose_loc  = tensor2np(model.pose_poredict_model(state["beliefs"])["loc"])
        predicted_pose_scale = tensor2np(model.pose_poredict_model(state["beliefs"])["scale"])

        input_pose = actions[0:-1].detach().cpu().numpy()
        reconstructions = {"image_hsr_256": reconstructions}
        # reconstructions = {"image_hsr_256": **reconstructions}
    # Get room ID for color map
    
    
    room_label = []
    for t in range(len(actions)-1):
        room_label.append(room_clustering(input_pose[t, 0]))
    room_label = np.stack(room_label)

    conversion_table = {
        0: "darkorange",
        1: "magenta",
        2: "blue",
        3: "red",
        4: "gold",
        5: "darkgreen",
        6: "purple",
        7: "lime",
        8: "cyan"
    }
    color_s = convert_elements(room_label, conversion_table)

    ht = state["beliefs"] #posterior mean (episod_length, 1, ht_size)
    st_q = state["posterior_means"]

    reduction="PCA" #PCA or t-SNE
    # Reduse demension of latent space
    if reduction=="PCA":
        # PCA
        flatten_ht = tensor2np(ht).reshape(-1, ht.shape[-1])
        ht_pca_3d = pca_ht_3d.transform(flatten_ht)
        ht_pca_2d = pca_ht_2d.transform(flatten_ht)

        flatten_st_q = tensor2np(st_q).reshape(-1, st_q.shape[-1])
        st_q_pca_3d = pca_st_q_3d.transform(flatten_st_q)
        st_q_pca_2d = pca_st_q_2d.transform(flatten_st_q)

        ht_2d = ht_pca_2d
        st_2d = st_q_pca_2d
        ht_3d = ht_pca_3d
        st_3d = st_q_pca_3d
        
    elif reduction=="t-SNE":
        #T-SNE
        tsne = TSNE(n_components = 2,random_state=0,init='pca')
        ht_tsne_2d = tsne.fit_transform(ht.detach().cpu().numpy()[:,0])
        st_q_tsne_2d = tsne.fit_transform(st_q.detach().cpu().numpy()[:,0])
        tsne = TSNE(n_components = 3,random_state=0,init='pca')
        ht_tsne_3d = tsne.fit_transform(ht.detach().cpu().numpy()[:,0])
        st_q_tsne_3d = tsne.fit_transform(st_q.detach().cpu().numpy()[:,0])

        ht_2d = ht_tsne_2d
        st_2d = st_q_tsne_2d
        ht_3d = ht_tsne_3d
        st_3d = st_q_tsne_3d
    else:
        raise NotImplementedError

    # Imagination
    t_max = len(actions)
    # t_imagination_start = 30
    t_imagination_start = t_max-2
    ht_img = [state["beliefs"][t_imagination_start]]
    st_q_img = [state["posterior_means"][t_imagination_start]]

    len_imagination = t_max-t_imagination_start
    for t in range(len_imagination):
        belief, _, prior_mean, _ = model.transition_model(st_q_img[t], actions[t_imagination_start+t].unsqueeze(dim=0), ht_img[t], det=True) #det=True means deterministic. i.e. no sampling, prior_mean is the same as prior_state
        ht_img.append(belief.squeeze(dim=0))
        st_q_img.append(prior_mean.squeeze(dim=0))

    ht_img = torch.stack(ht_img)
    st_q_img = torch.stack(st_q_img)

    reconstructions_img = model.observation_model(h_t=ht_img, s_t=st_q_img)

    #潜在空間表現(imgnation)(PCA)
    feat = tensor2np(ht_img).reshape(-1, ht_img.shape[-1])
    feat_pca = pca_ht_3d.transform(feat)
    ht_img_x, ht_img_y, ht_img_z = get_xyz(feat_pca)

    feat = tensor2np(st_q_img).reshape(-1, st_q_img.shape[-1])
    feat_pca = pca_st_q_3d.transform(feat)
    st_q_img_x, st_q_img_y, st_q_img_z = get_xyz(feat_pca)


    # plot fig format
    w_graph = 4
    h_graph = 2

    fig = plt.figure(figsize=(w_graph*5,h_graph*5))
    fig.set_tight_layout(True)
    ax1 = fig.add_subplot(h_graph, w_graph, 1)
    ax2 = fig.add_subplot(h_graph, w_graph, 2)
    ax3 = fig.add_subplot(h_graph, w_graph, 3)
    ax4 = fig.add_subplot(h_graph, w_graph, 4)
    ax5 = fig.add_subplot(h_graph, w_graph, 5)
    ax6 = fig.add_subplot(h_graph, w_graph, 6)
    ax7 = fig.add_subplot(h_graph, w_graph, 7, projection="3d")
    ax8 = fig.add_subplot(h_graph, w_graph, 8, projection="3d")


    image_key="image_hsr_256"
    pose_key="Pose"
    n_frame = len(reconstructions[image_key]["loc"])

    dt = 1
    size = (256,256)

    map_path = '/root/TurtleBot3/catkin_ws/src/rgiro_sweethome3d_worlds/maps/type_2/3ldk/3ldk_01/map.pgm'
    map_image , map_info = load_map(map_path)
    layout_image = plt.imread("./room_layout_2.png")

    def plot_fig(t):
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

        fig.suptitle("time t={}s ".format(round(t)))

        # ax1: plot observation image
        im = observations_target[image_key][t, 0][[2, 1, 0]].detach().cpu().numpy()
        im = reverse_image_observation(im)
        im = cv2.resize(im, size, interpolation=cv2.INTER_LINEAR)
        ax1.imshow(im)
        ax1.set_title("Image: Observation")
        
        # ax2: plot reconstruction image
        im = reconstructions[image_key]["loc"][t, 0][[2, 1, 0]].detach().cpu().numpy()
        im = reverse_image_observation(im)
        im = cv2.resize(im, size, interpolation = cv2.INTER_LINEAR)
        ax2.imshow(im)
        ax2.set_title("Image: Reconstruction")
        
        # ax3: plot h(t) in 2D
        ax3.scatter(ht_2d[0:t+1,0], ht_2d[0:t+1,1], color = color_s[0:t+1], label="h",s=10)

        ax3.set_xlim(math.floor(ht_2d[:,0].min()), math.ceil(ht_2d[:,0].max()))
        ax3.set_ylim(math.floor(ht_2d[:,1].min()), math.ceil(ht_2d[:,1].max()))
        ax3.set_title("h(t) 2D ({})".format(reduction))

        # ax4: plot s(t) in 2D
        ax4.scatter(st_2d[0:t+1,0], st_2d[0:t+1,1], color = color_s[0:t+1], label="s",s=10)
        ax4.set_xlim(math.floor(st_2d[:,0].min()), math.ceil(st_2d[:,0].max()))
        ax4.set_ylim(math.floor(st_2d[:,1].min()), math.ceil(st_2d[:,1].max()))
        ax4.set_title("s(t) 2D ({})".format(reduction))
        
        # ax5: plot map, poses and trajectory
        ax5.imshow(map_image, extent=(map_info['origin'][0],map_info['origin'][0]+map_image.size[0]*map_info['resolution'],map_info['origin'][1],map_info['origin'][1]+map_image.size[1]*map_info['resolution']), cmap='gray')
        ax5.set_xlim(-7,7)
        ax5.set_ylim(-7,7)
        
        pose_transformed, arrow_end = transform_pose(input_pose[0:t+1, 0])
        # pose_transformed, arrow_end = transform_pose(observations_target["Pose"][0:t+1, 0].detach().cpu().numpy())
        ax5.scatter(pose_transformed[t,0],pose_transformed[t,1],s=5,c="red")
        ax5.annotate('', xy=arrow_end, xytext=[pose_transformed[t,0],pose_transformed[t,1]],
                arrowprops=dict(shrink=0, width=1, headwidth=3, 
                                headlength=5, connectionstyle='arc3',
                                facecolor='red', edgecolor='red')
                )

        ax5.plot(pose_transformed[0:t,0],pose_transformed[0:t,1])

        pose_transformed, arrow_end = transform_pose(np.expand_dims(predicted_pose_loc[t,0],axis=0))
        ax5.scatter(pose_transformed[-1,0],pose_transformed[-1,1], s = 8, c="green")
        ax5.annotate('', xy=arrow_end, xytext=[pose_transformed[-1,0],pose_transformed[-1,1]],
            arrowprops=dict(shrink=0, width=1, headwidth=3, 
                            headlength=5, connectionstyle='arc3',
                            facecolor='red', edgecolor='green')
            )

        el_width,el_height,el_angle = get_ellipse(predicted_pose_scale[t,0,0],predicted_pose_scale[t,0,1])
        E = pat.Ellipse(xy=tuple(pose_transformed[-1,0:2]),
                        # width=predicted_pose_scale[t,0,0]*2*1.6*2,
                        # height=predicted_pose_scale[t,0,1]*2*1.6*2,
                        width=el_width,
                        height=el_height,
                        angle=el_angle,
                        color="lime", alpha=0.2)
        ax5.add_patch(E)
        if "Pose" in reconstructions.keys():
            ax5.set_title("Pose:\n Observation(red)  Reconstruction(green)")
        else:
            ax5.set_title("Pose:\n Action(red)  prediction(green)")

        # ax: plot layout image
        ax6.imshow(layout_image)
        ax6.set_title("Room layout")

        t_obs = t - t_imagination_start
        
        if t<t_imagination_start:
            # # ax6: plot imagination image
            # im = reconstructions[image_key]["loc"][t, 0][[2, 1, 0]].detach().cpu().numpy()
            # # im = recon["loc"][t, 0][[2,1,0]].detach().cpu().numpy()
            # im = reverse_image_observation(im)
            # im = cv2.resize(im, size, interpolation = cv2.INTER_LINEAR)
            # ax6.imshow(im)
            # ax6.set_title("Image: Reconstruction")
            # # pint(t)
            
            # ax7: plot h(t) in 3D
            # ax7.plot(ht_pca_x[0:t], ht_pca_y[0:t], ht_pca_z[0:t], color = color_s[0:t+1], alpha = 0.8)
            ax7.scatter(ht_3d[0:t+1,0], ht_3d[0:t+1,1], ht_3d[0:t+1,2], color = color_s[0:t+1], label = "h", alpha = 0.5)
            ax7.scatter(ht_3d[t,0], ht_3d[t,1], ht_3d[t,2], label = "h",color = color_s[t],s=8)
            ax7.view_init(20, t*3)
            ax7.set_xlim(math.floor(ht_3d[:,0].min()), math.ceil(ht_3d[:,0].max()))
            ax7.set_ylim(math.floor(ht_3d[:,1].min()), math.ceil(ht_3d[:,1].max()))
            ax7.set_zlim(math.floor(ht_3d[:,2].min()), math.ceil(ht_3d[:,2].max()))
            ax7.set_title("h(t) 3D ({})".format(reduction))

            # ax8: plot s(t) in 3D
            # ax8.plot(st_q_pca_x[0:t+1], st_q_pca_y[0:t+1], st_q_pca_z[0:t+1], alpha=0.8)
            ax8.scatter(st_3d[0:t+1,0], st_3d[0:t+1,1], st_3d[0:t+1,2], color = color_s[0:t+1], label = "s", alpha=0.5)
            ax8.scatter(st_3d[t,0], st_3d[t,1], st_3d[t,2], label="s",color = color_s[t],s=8)
            ax8.view_init(20, t*3)
            ax8.set_xlim(math.floor(st_3d[:,0].min()), math.ceil(st_3d[:,0].max()))
            ax8.set_ylim(math.floor(st_3d[:,1].min()), math.ceil(st_3d[:,1].max()))
            ax8.set_zlim(math.floor(st_3d[:,2].min()), math.ceil(st_3d[:,2].max()))
            ax8.set_title("s(t) 3D ({})".format(reduction))
        else:
            # ax5: plot imagination image
            reconstruction_pose = reconstructions_img[pose_key]["loc"][t_obs,0].detach().cpu().numpy()
            pose_transformed, arrow_end = transform_pose(reconstruction_pose)
            ax5.scatter(pose_transformed[0],pose_transformed[1], s = 8, c="green")
            ax5.annotate('', xy=arrow_end, xytext=[pose_transformed[0],pose_transformed[1]],
                arrowprops=dict(shrink=0, width=1, headwidth=3, 
                                headlength=5, connectionstyle='arc3',
                                facecolor='red', edgecolor='darkorange')
            )
            ax5.set_title("Pose:\n Observation(red) Reconstruction(green)\n Imagination(orange)")
            # ax5.set_title("Pose:\n Observation(red) Reconstruction(green) Imagination(orange)\n Room {}".format(room_label[t]))

            # # ax6: plot imagination image
            # im = reconstructions_img[image_key]["loc"][t_obs, 0][[2,1,0]].detach().cpu().numpy()
            # im = reverse_image_observation(im)
            # im = cv2.resize(im, size, interpolation=cv2.INTER_LINEAR)
            # ax6.imshow(im)
            # ax6.set_title("Image: Imagination")

            # ax7: plot h(t) in 3D        
            # ax7.plot(ht_img_x[t_imagination_start:t_obs], ht_img_y[t_imagination_start:t_obs], ht_img_z[t_imagination_start:t_obs], alpha=0.4)
            # ax7.scatter(ht_img_x[t_obs],ht_img_y[t_obs],ht_img_z[t_obs],label="h")
            # ax7.set_xlim(-10,10)
            # ax7.set_ylim(-10,10)
            # ax7.set_zlim(-10,10)
            # ax7.set_title("h(t): Imagination ")

            # # ax8: plot s(t) in 3D
            # ax8.plot(st_q_img_x[t_imagination_start:t_obs], st_q_img_y[t_imagination_start:t_obs], st_q_img_z[t_imagination_start:t_obs], alpha=0.4)
            # ax8.scatter(st_q_img_x[t_obs], st_q_img_y[t_obs], st_q_img_z[t_obs], label="s")
            # ax8.set_xlim(-30,30)
            # ax8.set_ylim(-30,30)
            # ax8.set_zlim(-30,30)
            # ax8.set_title("s(t):Imagination")


    # 4. アニメーション化
    anim = FuncAnimation(fig, plot_fig, tqdm(np.arange(n_frame)), interval= dt*333)
    # anim = FuncAnimation(fig, plot_fig, tqdm(np.arange(50)), interval= dt*333)

    os.makedirs(output_dir, exist_ok=True)
    import re
    pattern = r'ver\d'
    version = re.findall(pattern, os.path.basename(__file__))
    now = datetime.datetime.now()
    if data=="train":
        save_file_name = f"{output_dir}/train_ep{epi_idx}_{reduction}_{version[0]}_{now.strftime('%Y_%m%d_%H%M')}.mp4"
    else:
        save_file_name = "{}/{}_val_ep{}_{}_{}.mp4".format(output_dir,now.strftime('%Y%m%d%H%M'), epi_idx, reduction, version[0])

    anim.save(save_file_name, writer='ffmpeg')
    print(save_file_name)
    print("fin")

if __name__ == "__main__":
    main()
