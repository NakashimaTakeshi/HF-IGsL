#!/usr/bin/env python
import os
import sys
import numpy as np
import rosbag
import warnings
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ArtistAnimation
from tqdm import tqdm
import math


from record_utils import *


if __name__ == '__main__':
    target_hz = 1
    target_topics = dict(
        Pose=dict(
            topic_name = "/global_pose",
            topic_msg_type = "geometry_msgs/PoseStamped",
            buf_clear = True,
        ),
    )

    bag_path = sys.argv[1]
    in_bag = rosbag.Bag(bag_path)

    t_s = in_bag.get_start_time()
    t_e = in_bag.get_end_time()

    read_msg_list = []
    for key in target_topics.keys():
        read_msg_list.append(target_topics[key]["topic_name"])

    recoder = TopicRecoder(target_hz, target_topics)

    if "sound" in target_topics.keys():
        recoder.init_audio(in_bag)

    # collect topic
    for topic, msg, t in in_bag.read_messages(topics=read_msg_list):
        percentage = (t.to_sec() - t_s) / (t_e - t_s)
        sys.stdout.write('\r{0:4.2f}%'.format(100 * percentage))
        t = t.to_sec() - t_s
        recoder(topic, msg, t)
        
    sys.stdout.flush()
    recoder.obs_complement()

    out_path = os.path.splitext(bag_path)[0]+".npy"
    
    print(len(recoder.pose_list[:,3]))
    print(np.max(recoder.pose_list[:,3]))
    print(np.min(recoder.pose_list[:,3]))

    img = plt.imread("map_reshape_2.bmp")

    h_graph = 1
    w_graph = 1
    fig = plt.figure(figsize=(w_graph*5,h_graph*5))
    ax1 = fig.add_subplot(h_graph, w_graph, 1)

    
    # ax1.set_xlim(830,1220)
    # ax1.set_ylim(1150,900)

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    t_imag_start=20
    dt = 1
    artists = []
    long=30
    im = ax1.imshow(img)
    # for t in range(len(recoder.pose_list)):
    #     plt.cla()
    #     plt.xticks([])
    #     plt.yticks([])
    #     im1 = ax1.imshow(img)
    #     im2 = ax1.scatter(recoder.pose_list[t,0],recoder.pose_list[t,1],s=5,c="red")
    #     x_end=long*np.cos((recoder.pose_list[t,3]/180)*math.pi)+recoder.pose_list[t,0]
    #     y_end=-1*long*np.sin((recoder.pose_list[t,3]/180)*math.pi)+recoder.pose_list[t,1]
        
    #     im3 = ax1.annotate('', xy=[x_end,y_end], xytext=[recoder.pose_list[t,0],recoder.pose_list[t,1]],
    #             arrowprops=dict(shrink=0, width=1, headwidth=3, 
    #                             headlength=5, connectionstyle='arc3',
    #                             facecolor='red', edgecolor='red')
    #            )
    #     im4 = ax1.plot(recoder.pose_list[0:t,0],recoder.pose_list[0:t,1])
        
        
    #     ax1.set_title("Dataset")
    #     print(1)
    #     print(im1)
    #     print(2)
    #     print(im2)
    #     print(3)
    #     print(im3)
    #     print(4)
    #     print(im4)
    #     artists.append([im1]+[im2]+[im3]+im4)
    #     plt.pause(0.01)
        
    def plot(t):
        plt.cla()
        ax1.cla()
        fig.suptitle("time t={}".format(t))

        ax1.imshow(img)
        ax1.scatter(recoder.pose_list[t,0],recoder.pose_list[t,1],s=5,c="red")

        x_end=long*np.cos((recoder.pose_list[t,3]/180)*math.pi)+recoder.pose_list[t,0]
        y_end=-1*long*np.sin((recoder.pose_list[t,3]/180)*math.pi)+recoder.pose_list[t,1]
        ax1.annotate('', xy=[x_end,y_end], xytext=[recoder.pose_list[t,0],recoder.pose_list[t,1]],
                arrowprops=dict(shrink=0, width=1, headwidth=3, 
                                headlength=5, connectionstyle='arc3',
                                facecolor='red', edgecolor='red')
               )
        ax1.plot(recoder.pose_list[0:t,0],recoder.pose_list[0:t,1])
        

    # 4. アニメーション化
    # anim = ArtistAnimation(fig, tqdm(artists), interval=100*dt)
    anim = FuncAnimation(fig, plot, frames=len(recoder.pose_list), interval=100*dt)

    folder_name = ""

    #os.makedirs(folder_name, exist_ok=True)
    save_file_name = "val_10.mp4"

    anim.save(save_file_name, writer='ffmpeg')
    plt.close()


    print("fin")


    print('\nDone!')
