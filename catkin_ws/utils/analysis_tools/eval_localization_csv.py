#!/usr/bin/env python
# coding: utf-8
import sys
import os
# from pathlib import Path
import numpy as np
# from tqdm import tqdm
# import hydra
# from omegaconf import DictConfig, OmegaConf
# import torch
import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from hydra import initialize, initialize_config_module, initialize_config_dir, compose
# from omegaconf import OmegaConf
# import cv2
# import mpl_toolkits
import glob
# import math
# from sklearn.manifold import TSNE
# import datetime
# import matplotlib.patches as pat
import csv

# module_path = os.path.join(Path().resolve(), '../../../../..')
# sys.path.append(module_path)

# #pythonpathの追加
# os.environ['PYTHONPATH'] = module_path
# sys.path.append(os.path.join(Path().resolve(), '../Multimodal-RSSM'))


# from pyquaternion import Quaternion
# from scipy.spatial.transform import Rotation as R

#パラーメーター設定
# import glob
args = sys.argv
working_dir = args[1]
folders = sorted(glob.glob(os.path.join(working_dir, "Path*")))
print(folders)

save_data = [["Path", "Average", "std", "std_error", "Num", "平均(標準偏差)","best", "worst","median"]]

for folder in folders:
    file_dir = folder
    files = glob.glob(file_dir +"/*"+".npy")
    if len(files) == 0:
        continue
    data_np = np.load(files[0], allow_pickle=True).item()
    num = len(data_np["pose_t-1"])-5
    print(folder)

    culc_data = np.zeros((len(files), num))
    print("Num=",num)


    fig = plt.figure(facecolor="w")
    ax = fig.add_subplot(1, 1, 1)

# load data
    position_data={"pose_t-1":[],
                   "grand_pose_t":[]
                    }

    for file in range(len(files)):
        # file_path = os.path.join(file_dir, files[file]) 
        data_np = np.load(files[file], allow_pickle=True).item()
        position_data["pose_t-1"].append(data_np["pose_t-1"])
        position_data["grand_pose_t"].append(data_np["grand_pose_t"])
        for i in range(num):
            data_pose = data_np["pose_t-1"][i+1]
            data_grand_pose = data_np["grand_pose_t"][i]

            culc_data[file][i] = np.sqrt(np.mean((data_grand_pose - data_pose) ** 2))
        ax.plot(np.arange(num), culc_data[file], alpha=0.15,  lw=1)

    max_len = max(len(arr) for arr in position_data["pose_t-1"])
    min_len = min(len(arr) for arr in position_data["pose_t-1"])
    print("max_len", max_len, "   / min_len", min_len)
    mse_list=[]
    average_array=np.zeros(len(position_data["grand_pose_t"]))
    for j,(grand_truth , pose) in enumerate(zip(position_data["grand_pose_t"], position_data["pose_t-1"])):
        if len(grand_truth) != len(pose):
            print("error")
            print(len(grand_truth), len(pose))
            exit()
            
        mse_list.append([])
        for i in range(len(grand_truth)-1):
            # if i == 0:
            #     continue
            data_pose = pose[i+1]
            data_grand_pose = grand_truth[i]
            mse_list[j].append(np.sqrt(np.mean((data_grand_pose - data_pose) ** 2)))
            # culc_data[file][i-1] = np.sqrt(np.mean((data_grand_pose - data_pose) ** 2))
        average_array[j]= sum(mse_list[j]) / len(mse_list[j])
        ax.plot(np.arange(len(mse_list[j])), mse_list[j], alpha=0.15,  lw=1)

    # average_array
    # average_array = culc_data.mean(axis=1)
    # final_pose_error = culc_data[:,-1]
    final_pose_error = [sublist[-1] for sublist in mse_list if sublist]
    best_ave_episode = np.argmin(average_array)
    worst_ave_episode = np.argmax(average_array)
    median_ave_episode = np.argsort(average_array)[int(len(average_array)/2)]
    best_final_episode = np.argmin(final_pose_error)
    worst_final_episode = np.argmax(final_pose_error)
    median_final_episode = np.argsort(final_pose_error)[int(len(final_pose_error)/2)]
    
    average = average_array.mean(axis = 0)
    std = np.std(average_array, ddof=1) 
    std_error = np.std(average_array, ddof=1) / np.sqrt(len(average_array))

    print("Average",round(average,2))
    print("std:", round(std,2))
    print("std_error", round(std_error, 2))

    save_data.append([os.path.basename(folder), average, std, std_error, num, str(round(average,2))+'±'+str(round(std,2)),best_final_episode, worst_final_episode, median_final_episode,best_ave_episode, worst_ave_episode, median_ave_episode])


with open(os.path.join(working_dir, "save_data.csv"), 'w') as f:
  writer = csv.writer(f)
  writer.writerows(save_data)

f.close()
