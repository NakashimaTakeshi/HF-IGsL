#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from math import pi as PI

import actionlib
import rospy
import actionlib_msgs.msg as actionlib_msgs
import geometry_msgs.msg as geometry_msgs
import numpy as np
from tf.transformations import quaternion_from_euler

from geometry_msgs.msg import Quaternion
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal


class WaypointNavigation():
    def __init__(self, action_topic="/move_base", goal_position_topic="/debug/goal_pose", world="aws_robomaker_small_house_world", each_area_point_number=10):
        # rospy.init_node("waypoint_navigation", anonymous=True)
        # Set action client
        self.cli = actionlib.SimpleActionClient(action_topic, MoveBaseAction)
        # Set publisher node
        self.goal_point_pub = rospy.Publisher(goal_position_topic, geometry_msgs.PoseStamped, queue_size=10)
        self.poses = self.read_pose_from_csv_file(world, each_area_point_number)

    def execute(self, pose):
        # Set "action_goal"
        action_goal = MoveBaseGoal()

        # Set "target_pose"
        target_pose = self.set_target_pose(pose)
        rospy.sleep(1)
        action_goal.target_pose = target_pose

        # Publish "target_pose"
        self.goal_point_pub.publish(target_pose)

        # Send action goal
        self.cli.send_goal(action_goal)
        self.cli.wait_for_result()

        if self.cli.get_state() == actionlib_msgs.GoalStatus.SUCCEEDED:
            rospy.loginfo("Move succeeded. ")
        else:
            rospy.logerr("Move failed. " + "failed pose: " + str(pose) + ".")

    def read_pose_from_csv_file(self, world, number):
        area_centroid_pose = []
        poses = []
        for line in open( '../waypoints/' + world + '.csv', 'r'):
            pose = line[:-1].split(',')
            if not (pose[0].startswith('#') or line==('\n')):
                pose_float = [float(item) for item in pose]
                area_centroid_pose.append(pose_float)
        for i in range(len(area_centroid_pose)):
            # rospy.loginfo("The centroid pose of the current area is " + str(area_centroid_pose[i]) + ".")
            pose = self.generate_multi_pose_from_each_area(area_centroid_pose[i], number)
            poses.append(pose)
        return poses

    def generate_multi_pose_from_each_area(self, pose, number):
        poses = []
        for i in range(number):
            tmp_pose = pose + np.random.randn(1, 4) * 0.5
            tmp_pose[0][2] = 0.0
            # rospy.loginfo("The multi pose of the current area is " + str(tmp_pose[0]) + ".")
            poses.append(tmp_pose[0])
        return poses

    def set_target_pose(self, pose): 
        target_pose = geometry_msgs.PoseStamped()
        target_pose.header.frame_id = 'map'
        target_pose.header.stamp = rospy.Time.now()
        target_pose.pose.position.x = pose[0]
        target_pose.pose.position.y = pose[1]
        target_pose.pose.position.z = pose[2]
        target_pose.pose.orientation = self.euler_to_quaternion(pose[3])
        return target_pose

    def euler_to_quaternion(self, orientation_z):
        radian = orientation_z * (PI / 180)
        q = quaternion_from_euler(0., 0., radian)
        return Quaternion(q[0], q[1], q[2], q[3])

if __name__=="__main__":
    waypointNavigation = WaypointNavigation()
    # waypointNavigation.execute()
