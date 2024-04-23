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

# module_path = os.path.join(Path().resolve(), '../../../../..')
# sys.path.append(module_path)

#pythonpathの追加
# os.environ['PYTHONPATH'] = module_path
# sys.path.append(os.path.join(Path().resolve(), '../Multimodal-RSSM'))




#パラーメーター設定
# import glob
args = sys.argv
file_dir = args[1]
# files = glob.glob("./"+ file_dir +"/*"+".npy")
files = glob.glob(file_dir +"/*"+".npy")
data_np = np.load(files[0], allow_pickle=True).item()
# Check if any .npy files were found
if not files:
    print(f"Error: No .npy files found in the directory '{file_dir}'.")
    sys.exit(1)

num = len(data_np["pose_t-1"])-2
print(num)

culc_data = np.zeros((len(files), num))
print("Num=",num)

fig = plt.figure(facecolor="w")
ax = fig.add_subplot(1, 1, 1)


for file in range(len(files)):
    # file_path = os.path.join(file_dir, files[file]) 
    data_np = np.load(files[file], allow_pickle=True).item()
    # print(files[file])
    # print(data_np.keys())
    print(files[file])
    print(f"len:{len(data_np['pose_t-1'])})")
    for i in range(num):
        data_pose = data_np["pose_t-1"][i+1]
        data_grand_pose = data_np["grand_pose_t"][i+1]
        # data_grand_pose = data_np["grand_pose_t"][i]

        culc_data[file][i] = np.sqrt(np.mean((data_grand_pose - data_pose) ** 2))
    ax.plot(np.arange(num), culc_data[file], alpha=0.15,  lw=1)

average_array = culc_data.mean(axis=1)
average = average_array.mean(axis = 0)
std = np.std(average_array, ddof=1) 
std_error = np.std(average_array, ddof=1) / np.sqrt(len(average_array))

print("Average",round(average,2))
print("std:", round(std,2))
print("std_error", round(std_error, 2))

script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
path_name = os.path.basename(file_dir)

ax.set_title(path_name)
ax.set_xlabel("Step")
ax.set_ylabel("Root Mean Squared Error[m]")
ax.set_ylim(0,7)
ax.plot(np.arange(len(culc_data.mean(axis=0))),culc_data.mean(axis=0), color = "red", lw=3, label="Average")
result = 'Time average error='+str(round(np.average(culc_data.mean(axis=0)),2))
ax.text(0.99, 0.99, result, va='top', ha='right', transform=ax.transAxes)
ax.legend(loc="upper left")
plt.savefig(os.path.join(script_dir, path_name + "_graph.pdf"))
plt.savefig(os.path.join(script_dir, "mcl_" + path_name + "_graph.png"), dpi=400)
