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
from torch.distributions import Normal

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
import tf
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, Point, Quaternion
class RSSM_ros():
    def __init__(self):
        # 勾配を計算しない
        torch.set_grad_enabled(False)

        # #パラーメーター設定
        path_name = "HF-PGM_model2-seed_0/2023-01-26/run_2"
        model_idx = 2
        cfg_device = "cuda:0"

        # 相対パス（ここを変えれば、コマンドを実行するdirを変更できる、必ず"config_path"は相対パスで渡すこと！！）
        model_folder =  os.path.join("./../Multimodal-RSSM/train/HF-PGM/House/MRSSM/MRSSM/results", path_name) 
        #launch用のパス（絶対パス）
        model_folder_launch =  os.path.join("../TurtleBot3/catkin_ws/src/ros_rssm/Multimodal-RSSM/train/HF-PGM/House/MRSSM/MRSSM/results", path_name) 

        # print("model_folder",model_folder)
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

        # # load states
        # state_path = model_path.replace("models", "states_models").replace(".pth", ".npy")
        # print("state_path:", state_path)

        # states_np = np.load(state_path, allow_pickle=True).item()
        # print("-- dataset --")
        # for key in states_np.keys():
        #     print(key)

        # print("-- key of states --")
        # print(states_np[key].keys())

        self.pose_predict_loc = []
        self.pose_predict_scale = []
        self.past_belief = torch.zeros(1, self.model.cfg.rssm.belief_size, device=self.model.cfg.main.device)
        self.past_state_without_pose = torch.zeros(1, self.model.cfg.rssm.state_size, device=self.model.cfg.main.device)
        self.pose_dummy = torch.zeros((1, 1, self.model.cfg.env.observation_shapes["Pose"][0]), device=self.model.cfg.main.device)
        self.i=1

        self.eval_data = dict()
        eval_data_key = ["image_t", "pose_t-1", "grand_pose_t","predict_pose_loc", "predict_pose_scale","past_posterior_states","past_belief" ,"posterior_states_without_pose","belief_without_pose", "past_state_prams","state_without_pose_prams"]

        for key in eval_data_key:  
            self.eval_data[key] = []
        
        self.mode = True

        now = datetime.datetime.now()
        args = sys.argv
        filename = './../TurtleBot3/ex_data/JSAI/log_model2_particle_'+ args[1] + now.strftime('%Y%m%d_%H%M%S') + '.npy'
        self.out_path = filename


    def ros_init_(self):
        rospy.init_node('PredictPosition_RSSM_server')
        self.pose_subscriber()
        self.image_subscriber()
        self.grand_pose_subscriber()
        self.PredictPosition_RSSM_server()
        self.PredictPosition_RSSM_publisher()
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

        # if os.path.exists(os.path.dirname(self.out_path)) == False:
        #     os.mkdir(os.path.dirname(self.out_path))

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

    # def quaternion2euler_numpy(self, x, y, z, w):
    #     """
    #     Convert a quaternion into euler angles (roll, pitch, yaw)
    #     roll is rotation around x in radians (counterclockwise)
    #     pitch is rotation around y in radians (counterclockwise)
    #     yaw is rotation around z in radians (counterclockwise)
    #     """
    #     t0 = +2.0 * (w * x + y * z)
    #     t1 = +1.0 - 2.0 * (x * x + y * y)
    #     roll_x = np.degrees(np.arctan2(t0, t1))
    #     t2 = +2.0 * (w * y - z * x)
    #     t2 = np.where(t2>+1.0,+1.0,t2)
    #     t2 = np.where(t2<-1.0, -1.0, t2)
    #     pitch_y = np.degrees(np.arcsin(t2))
    #     t3 = +2.0 * (w * z + x * y)
    #     t4 = +1.0 - 2.0 * (y * y + z * z)
    #     yaw_z = np.degrees(np.arctan2(t3, t4))
    #     return roll_x, pitch_y, yaw_z # in radians
    
    def quaternion_to_euler(self, quaternion):
        """Convert Quaternion to Euler Angles

        quarternion: geometry_msgs/Quaternion
        euler: geometry_msgs/Vector3
        """
        e = tf.transformations.euler_from_quaternion((quaternion[1], quaternion[1], quaternion[2], quaternion[3]))
        return e



    def PredictPosition_RSSM(self, req):
        if self.i > 1:
            sub_data = dict(image=self.img.transpose(2, 0, 1), pose=self.pose, grand_pose=self.grand_pose_receiver)
        else:
            sub_data = dict(image=self.img.transpose(2, 0, 1), grand_pose=self.grand_pose_receiver)

        normalized_img = (normalize_image(np2tensor(sub_data["image"]), 5).unsqueeze(0).unsqueeze(0).to(device=self.model.device))
        action = torch.zeros([1, 1, 1], device=self.model.device)

        if "pose" in sub_data.keys():  # 時刻t=0では動かないような条件
            predict_pose = np2tensor(sub_data["pose"]).unsqueeze(0).unsqueeze(0).to(device=self.model.device)
            # predict_pose = np2tensor(sub_data["grand_pose"]).unsqueeze(0).unsqueeze(0).to(device=self.model.device)

            normalized_past_img = (normalize_image(np2tensor(self.past_image), 5).unsqueeze(0).unsqueeze(0).to(device=self.model.device))
            past_observations_seq = dict(image_hsr_256=normalized_past_img, Pose=predict_pose)

            # # 1. s^q_t-1~q(s^q_t-1|h_t-1, s_t-1,o_t-1,x_t-1) 全ての観測から現在の状態を推論
            # past_state = self.model.estimate_state_online(
            #     past_observations_seq, action, self.past_state_without_pose, self.past_belief
            # )

            # 1. s^q_t-1~q(s^q_t-1|h_t-1, o_t-1,x_t-1) 全ての観測から1時刻前の状態を推論
            obs_emb_posterior = self.model.get_obs_emb_posterior(past_observations_seq)
            past_state_loc_and_scale = self.model.transition_model.obs_encoder(
                h_t=self.past_belief, o_t=obs_emb_posterior[0])
            past_state = Normal(past_state_loc_and_scale['loc'], past_state_loc_and_scale['scale']).rsample()
        else:
            past_state = self.past_state_without_pose

        # 2. h_t=f(s^q_t-1,h_t-1)
        print("past_state.shape", past_state.shape)
        # 3. s_t~q(s_t|h_t,o_t) imageだけで状態推論
        observations_seq = dict(
            image_hsr_256=normalized_img, Pose=self.pose_dummy
        )  # Poseは状態の推論には使用しないが、何らかの値がないとエラーが出る
        state = self.model.estimate_state_online(observations_seq, action, past_state, self.past_belief, subset_index=1)


        
        # 4. x_t~p(x_t|s_t,h_t) imageだけで推論した状態からxtを予測
        locandscale = self.model.observation_model.observation_models["Pose"](
            s_t=state["posterior_states"], h_t=state["beliefs"]
        )


        self.pose_predict_loc.append(tensor2np(locandscale["loc"])[0][0].tolist())
        self.pose_predict_scale.append(tensor2np(locandscale["scale"])[0][0].tolist())

        
        resp = SendRssmPredictPositionResponse()

        resp.x_loc = self.pose_predict_loc[-1][0]
        resp.y_loc = self.pose_predict_loc[-1][1]
        resp.cos_loc = self.pose_predict_loc[-1][2]
        resp.sin_loc = self.pose_predict_loc[-1][3]
        resp.x_scale = self.pose_predict_scale[-1][0]
        resp.y_scale = self.pose_predict_scale[-1][1]
        resp.cos_scale = self.pose_predict_scale[-1][2]*10
        resp.sin_scale = self.pose_predict_scale[-1][3]*10
        resp.weight = min(0.4, (1/1000)* (self.i - 1)**2)
        resp.integration_mode = 2.0
        print(resp.weight)
        # resp.weight = 0


        if self.mode == True:
            print("---------------------------")
            print("HF-PGM (RSSM SERBER) | t = ", self.i)
            if self.i != 1:
                print("x_{t-1}        :", np.round(sub_data["pose"], decimals=2).tolist()) 
            print("p(x_t|h_t)[loc]:", [round(t, 2) for t in self.pose_predict_loc[-1]])
            print("Grand x_t      :", np.round(sub_data["grand_pose"], decimals=2).tolist())
            

    # "predict_pose_scale","beliefs", "posterior_means", "recon_posterior_means"
            
            self.eval_data["image_t"].append(sub_data["image"])
            if self.i != 1:
                self.eval_data["pose_t-1"].append(sub_data["pose"])
                self.eval_data["past_state_prams"].append(dict(loc=tensor2np(past_state_loc_and_scale["loc"]),scale=tensor2np(past_state_loc_and_scale["scale"])))
            else:
                self.eval_data["pose_t-1"].append(np.array([1000,1000,1,1]))
                self.eval_data["past_state_prams"].append(dict(loc=np.array([0]),scale=np.array([0])))
            self.eval_data["grand_pose_t"].append(sub_data["grand_pose"])
            self.eval_data["predict_pose_loc"].append(np.array(self.pose_predict_loc[-1]))
            self.eval_data["predict_pose_scale"].append(np.array(self.pose_predict_scale[-1]))
            self.eval_data["past_posterior_states"].append(tensor2np(past_state))
            self.eval_data["past_belief"].append(tensor2np(self.past_belief))
            self.eval_data["posterior_states_without_pose"].append(tensor2np(state["posterior_states"]))
            self.eval_data["belief_without_pose"].append(tensor2np(state["beliefs"]))
            
            self.eval_data["state_without_pose_prams"].append(dict(loc=tensor2np(state["posterior_means"]), scale = tensor2np(state["posterior_std_devs"])))



            # 更新処理
            self.past_image = sub_data["image"]
            self.past_belief, self.past_state_without_pose = state["beliefs"][0], state["posterior_states"][0]

            self.i += 1
            self.pub_rssm_predict_position.publish(resp2PoseWithCovariance(resp))


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

def resp2PoseWithCovariance(resp):
    euler = np.arctan2(resp.cos_loc,resp.sin_loc)
    rot = tf.transformations.quaternion_from_euler(0, 0, euler)

    rssm_estimate_pose = PoseWithCovarianceStamped()
    rssm_estimate_pose.header.stamp = rospy.get_rostime()
    rssm_estimate_pose.header.frame_id = "map"
    rssm_estimate_pose.pose.pose = Pose(Point(resp.x_loc, resp.y_loc, 0.0), Quaternion(rot[0], rot[1], rot[2], rot[3]))
    print("Quaternion :", rot[0], rot[1], rot[2], rot[3])
    rssm_estimate_pose.pose.covariance = [resp.x_scale,0.0,0.0,0.0,0.0,0.0,
                                          0.0,resp.y_scale,0.0,0.0,0.0,0.0,
                                          0.0,0.0,0.00000,0.0,0.0,0.0,
                                          0.0,0.0,0.0,0.00000,0.0,0.0,
                                          0.0,0.0,0.0,0.0,0.00000,0.0,
                                          0.0,0.0,0.0,0.0,0.0,0.0001]
    return rssm_estimate_pose


if __name__ == "__main__":
    
    test = RSSM_ros()
    test.ros_init_()