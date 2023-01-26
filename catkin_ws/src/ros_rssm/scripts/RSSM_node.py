#!/usr/bin/env python3
# coding: utf-8
import sys
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import hydra
import torch
import matplotlib.pyplot as plt
from hydra import initialize, compose
import glob
from scipy.spatial.transform import Rotation as R
import cv2
import datetime
import tf


# sys.path.append(os.path.join(Path().resolve(), 'catkin_ws/src/ros_rssm/Multimodal-RSSM'))
sys.path.append(os.path.join(Path().resolve(), '../TurtleBot3/catkin_ws/src/ros_rssm/Multimodal-RSSM'))
sys.path.append(os.path.join(Path().resolve(), '../Multimodal-RSSM'))
from algos.MRSSM.MRSSM.algo import build_RSSM
from algos.MRSSM.MRSSM.train import get_dataset_loader
from utils.evaluation.estimate_states import get_episode_data
from utils.evaluation.visualize_utils import get_pca_model, tensor2np, get_xy, get_xyz, reverse_image_observation

from ros_rssm.srv import *
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from std_srvs.srv import *
class RSSM_ros():
    def __init__(self):
        # 勾配を計算しない
        torch.set_grad_enabled(False)

        # #パラーメーター設定
        path_name = "HF-PGM_model1-seed_0/2023-01-25/run_3"
        model_idx = 2
        cfg_device = "cuda:0"

        # 相対パス（ここを変えれば、コマンドを実行するdirを変更できる、必ず"config_path"は相対パスで渡すこと！！）
        model_folder =  os.path.join("./../Multimodal-RSSM/train/HF-PGM/House/MRSSM/MRSSM/results", path_name) 
        #launch用のパス（絶対パス）
        model_folder_launch =  os.path.join("../TurtleBot3/catkin_ws/src/ros_rssm/Multimodal-RSSM/train/HF-PGM/House/MRSSM/MRSSM/results", path_name) 

        with initialize(config_path=model_folder):
            cfg = compose(config_name="hydra_config")

        # #仮の修正
        cfg.main.wandb=False

        cfg.main.device = cfg_device
        # # cfg.train.n_episode = 100
        print(' ' * 26 + 'Options')
        for k, v in cfg.items():
            print(' ' * 26 + k + ': ' + str(v))

        device = torch.device(cfg.main.device)

        # # # Load Model, Data and States
        model_paths = glob.glob(os.path.join(model_folder_launch, '*.pth'))
        print(model_paths)
        print("model_pathes: ")

        self.model = build_RSSM(cfg, device)
        model_path = model_paths[model_idx]
        self.model.load_model(model_path)
        self.model.eval()

 

        self.pose_predict_loc = []
        self.pose_predict_scale = []
        self.past_belief = torch.zeros(1, self.model.cfg.rssm.belief_size, device=self.model.cfg.main.device)
        self.past_state = torch.zeros(1, self.model.cfg.rssm.state_size, device=self.model.cfg.main.device)
        self.i=1

        self.eval_data = dict()
        eval_data_key = ["image_t", "pose_t-1", "grand_pose_t","predict_pose_loc", "predict_pose_scale","posterior_states","belief" ,"past_state_prams"]

        for key in eval_data_key:  
            self.eval_data[key] = []
        
        self.mode = True

        now = datetime.datetime.now()
        args = sys.argv
        filename = './../TurtleBot3/ex_data/log_model1_'+ args[1] + now.strftime('%Y%m%d_%H%M%S') + '.npy'
        self.out_path = filename




    def ros_init_(self):
        rospy.init_node('PredictPosition_RSSM_server')
        self.pose_subscriber()
        self.image_subscriber()
        self.grand_pose_subscriber()
        self.PredictPosition_RSSM_server()
        self.save_data_subscriber()
        rospy.on_shutdown(self.save_eval_data)
        print("Ready to PredictPosition_RSSM.")
        rospy.spin()

    def pose_subscriber(self):
        rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.callback_pose)

    def image_subscriber(self):
        rospy.Subscriber("/camera/rgb/image_raw/compressed", CompressedImage, self.callback_image)

    def grand_pose_subscriber(self):
        rospy.Subscriber("/tracker", Odometry, self.callback_grand_pose)

    def PredictPosition_RSSM_publisher(self):
        self.pub_rssm_predict_position = rospy.Publisher("PredictPosition_RSSM_topic", PoseWithCovarianceStamped)

    def save_data_subscriber(self):
        rospy.Service("Save_eval_data", Empty, self.save_eval_data)

    def save_eval_data(self):
        self.mode = False

        if os.path.exists(os.path.dirname(self.out_path)) == False:
            os.mkdir(os.path.dirname(self.out_path))

        np.save(self.out_path, self.eval_data, allow_pickle=True, fix_imports=True)
        print("Save eval data Dir=:", self.out_path)
        
        # return EmptyResponse()

    def callback_image(self, msg):
        img_cv2 = self.image_cmp_msg2opencv(msg)
        img_cut = img_cv2[ :, 80:560]
        self.img = cv2.resize(img_cut,(256, 256))
        # print(self.img.shape)
        # print(self.pose)

    def callback_pose(self, msg):
        self.pose = self.posewithcovariancestamped_converter(msg)
    
    def callback_grand_pose(self, msg):
        self.grand_pose_receiver = self.posewithcovariancestamped_converter(msg)


    def image_cmp_msg2opencv(self, image_msg):
        image_np = np.fromstring(image_msg.data, dtype=np.uint8)
        image_np = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        return image_np

    def posewithcovariancestamped_converter(self, msg):
        pose_list = self.pose_converter(msg.pose.pose)
        pose_list_oira = self.quaternion_to_euler([pose_list[3], pose_list[4], pose_list[5], pose_list[6]])
        pose_data = [pose_list[0], pose_list[1], np.cos(pose_list_oira[2]), np.sin(pose_list_oira[2])]
        return np.array(pose_data)
    
    def pose_converter(self, msg):
        position_list = self.geometry_msgs_vector3d_converter(msg.position)
        orientation_list = self.geometry_msgs_quaternion_converter(msg.orientation)
        return position_list + orientation_list

    def geometry_msgs_vector3d_converter(self, msg):
        return [msg.x, msg.y, msg.z]

    def geometry_msgs_quaternion_converter(self, msg):
        return [msg.x, msg.y, msg.z, msg.w]

    def quaternion_to_euler(self, quaternion):
        """Convert Quaternion to Euler Angles

        quarternion: geometry_msgs/Quaternion
        euler: geometry_msgs/Vector3
        """
        e = tf.transformations.euler_from_quaternion((quaternion[1], quaternion[1], quaternion[2], quaternion[3]))
        return e


    def PredictPosition_RSSM(self, req):
        sub_data = dict(image = self.img.transpose(2, 0, 1), pose = self.pose, grand_pose = self.grand_pose_receiver)
        print(sub_data["image"].shape)
        normalized_img = normalize_image(np2tensor(sub_data["image"]), 5).unsqueeze(0).unsqueeze(0).to(device=self.model.device)
        action = np2tensor(sub_data["pose"]).unsqueeze(0).unsqueeze(0).to(device=self.model.device)

        observations_seq = dict(image_hsr_256 = normalized_img)
        state = self.model.estimate_state_online(observations_seq, action, self.past_state, self.past_belief)

        self.past_belief, self.past_state = state["beliefs"][0], state["posterior_states"][0]
        locandscale = self.model.pose_poredict_model(self.past_belief)
        self.pose_predict_loc.append(tensor2np(locandscale["loc"])[0].tolist())
        self.pose_predict_scale.append(tensor2np(locandscale["scale"])[0].tolist())

        resp = SendRssmPredictPositionResponse()

        resp.x_loc = self.pose_predict_loc[-1][0]
        resp.y_loc = self.pose_predict_loc[-1][1]
        resp.cos_loc = self.pose_predict_loc[-1][2]
        resp.sin_loc = self.pose_predict_loc[-1][3]
        resp.x_scale = self.pose_predict_scale[-1][0]
        resp.y_scale = self.pose_predict_scale[-1][1]
        resp.cos_scale = self.pose_predict_scale[-1][2]*4
        resp.sin_scale = self.pose_predict_scale[-1][3]*4
        resp.weight = min(0.1, 0.01* (self.i - 1))

        if self.mode == True:
            print("---------------------------")
            print("HF-PGM (RSSM SERBER) | t = ", self.i)
            print("x_{t-1}        :", np.round(sub_data["pose"], decimals= 2).tolist())
            print("p(x_t|h_t)[loc]:", [round(t, 2) for t in self.pose_predict_loc[-1]])
            print("Grand x_t      :", np.round(sub_data["grand_pose"], decimals= 2).tolist())
            print("Grand x_t_now  :", np.round(self.grand_pose_receiver, decimals= 2).tolist())
            self.i += 1

            self.eval_data["image_t"].append(sub_data["image"])
            self.eval_data["pose_t-1"].append(sub_data["pose"])
            self.eval_data["grand_pose_t"].append(sub_data["grand_pose"])
            self.eval_data["predict_pose_loc"].append(np.array(self.pose_predict_loc[-1]))
            self.eval_data["predict_pose_scale"].append(np.array(self.pose_predict_scale[-1]))
            self.eval_data["posterior_states"].append(tensor2np(state["posterior_states"]))
            self.eval_data["belief"].append(tensor2np( state["beliefs"]))
            self.eval_data["past_state_prams"].append(dict(loc=tensor2np(state["posterior_means"]), scale = tensor2np(state["posterior_std_devs"])))

        else:
            print("fin")
            quit()
        return resp


    def PredictPosition_RSSM_server(self):
        s = rospy.Service('PredictPosition_RSSM', SendRssmPredictPosition, self.PredictPosition_RSSM)
        

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



if __name__ == "__main__":
    
    test = RSSM_ros()
    test.ros_init_()