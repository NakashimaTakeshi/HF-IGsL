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


        self.i=1

        self.eval_data = dict()
        eval_data_key = ["image_t", "pose_t-1", "grand_pose_t"]

        for key in eval_data_key:  
            self.eval_data[key] = []
        
        self.mode = True

        now = datetime.datetime.now()
        filename = 'amcl_log_' + now.strftime('%Y%m%d_%H%M%S') + '.npy'

        #ここのパスに保存されます。ファイル名は日時が入る。
        self.out_path = os.path.join("eval_data/amcl/dataset3", filename)




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
        pose_data = [pose_list[0], pose_list[1], np.cos(pose_list_oira[2]), np.sin(pose_list_oira[2], )]
        return np.array(pose_data)
    
    def pose_converter(self, msg):
        position_list = self.geometry_msgs_vector3d_converter(msg.position)
        orientation_list = self.geometry_msgs_quaternion_converter(msg.orientation)
        return position_list + orientation_list

    def geometry_msgs_vector3d_converter(self, msg):
        return [msg.x, msg.y, msg.z]

    def geometry_msgs_quaternion_converter(self, msg):
        return [msg.x, msg.y, msg.z, msg.w]

    def quaternion2euler_numpy(self, x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.degrees(np.arctan2(t0, t1))
        t2 = +2.0 * (w * y - z * x)
        t2 = np.where(t2>+1.0,+1.0,t2)
        t2 = np.where(t2<-1.0, -1.0, t2)
        pitch_y = np.degrees(np.arcsin(t2))
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.degrees(np.arctan2(t3, t4))
        return roll_x, pitch_y, yaw_z # in radians
    


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
        
        resp = SendRssmPredictPositionResponse()

        if self.mode == True:
            print("---------------------------")
            print("HF-PGM (AMCL_RECORD SERBER) | t = ", self.i)
            if self.i != 1:
                print("x_(t-1)        :{}   {}".format( np.round(sub_data["pose"], decimals=2).tolist(), np.degrees(np.arctan2(sub_data["pose"][3], sub_data["pose"][2]))))
            print("Grand x_t      :{}   {}".format(np.round(sub_data["grand_pose"], decimals=2).tolist(), np.degrees(np.arctan2(sub_data["grand_pose"][3], sub_data["grand_pose"][2]))))
            

    # "predict_pose_scale","beliefs", "posterior_means", "recon_posterior_means"
            
            self.eval_data["image_t"].append(sub_data["image"])
            if self.i != 1:
                self.eval_data["pose_t-1"].append(sub_data["pose"])
            else:
                self.eval_data["pose_t-1"].append(np.array([1000,1000,1,1]))

            self.eval_data["grand_pose_t"].append(sub_data["grand_pose"])

            self.i += 1


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