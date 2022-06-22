#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from math import pi as PI

import actionlib
import rospy
import actionlib_msgs.msg as actionlib_msgs
import geometry_msgs.msg as geometry_msgs
from tf.transformations import quaternion_from_euler

from geometry_msgs.msg import Quaternion
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal


def main():
    rospy.init_node("waypoint_navigation_2D_Nav_Goal", anonymous=True)
    #Move base goal
    action_topic="/move_base"
    goal_position_topic="/debug/goal_pose"

    pose = []
    pose.append([-2.2, -2.9956976240955777, 0.0])
    pose.append([360.0])

    # Set "target_pose"
    target_pose = set_target_pose(pose)

    # Set action client
    cli = actionlib.SimpleActionClient(action_topic, MoveBaseAction)
    rospy.sleep(1)
    # Set "action_goal"
    action_goal = MoveBaseGoal()
    action_goal.target_pose = target_pose

    # Set publisher node
    goal_point_pub = rospy.Publisher(goal_position_topic, geometry_msgs.PoseStamped, queue_size=10)
    goal_point_pub.publish(target_pose)
    
    # Send action goal
    cli.send_goal(action_goal)
    cli.wait_for_result()
    
    if cli.get_state() == actionlib_msgs.GoalStatus.SUCCEEDED:
        print("move succeeded.")
        return "succeeded"
    else:
        print("move failed.")
        cli.cancel_all_goals()
        return "failed"

def set_target_pose(pose): 
    target_pose = geometry_msgs.PoseStamped()
    target_pose.header.frame_id = 'map'
    target_pose.header.stamp = rospy.Time.now()
    target_pose.pose.position.x = pose[0][0]
    target_pose.pose.position.y = pose[0][1]
    target_pose.pose.position.z = pose[0][2]
    target_pose.pose.orientation = euler_to_quaternion(pose[1][0])
    return target_pose

def euler_to_quaternion(orientation_z):
    radian = orientation_z * (PI / 180)
    q = quaternion_from_euler(0., 0., radian)
    return Quaternion(q[0], q[1], q[2], q[3])

if __name__=="__main__":
    main()
