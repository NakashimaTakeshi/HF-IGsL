#!/usr/bin/env python3
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
    world = "aws_robomaker_small_house_world"

    poses = read_pose_from_csv_file(world)

    # Set action client
    cli = actionlib.SimpleActionClient(action_topic, MoveBaseAction)
    # Set "action_goal"
    action_goal = MoveBaseGoal()
    # Set publisher node
    goal_point_pub = rospy.Publisher(goal_position_topic, geometry_msgs.PoseStamped, queue_size=10)

    for pose in poses:
        # Set "target_pose"
        target_pose = set_target_pose(pose)
        rospy.sleep(1)
        action_goal.target_pose = target_pose

        # Publish "target_pose"
        goal_point_pub.publish(target_pose)

        # Send action goal
        cli.send_goal(action_goal)
        cli.wait_for_result()

        if cli.get_state() == actionlib_msgs.GoalStatus.SUCCEEDED:
            print("move succeeded. " + "current pose: " + str(pose))
        else:
            print("move failed.")
            cli.cancel_all_goals()

def read_pose_from_csv_file(world): 
    poses = []
    for line in open( '../waypoints/' + world + '.csv', 'r'):
        pose = line[:-1].split(',')
        if not (pose[0].startswith('#') or line==('\n')):
            pose_float = [float(item) for item in pose]
            poses.append(pose_float)
    return poses;

def set_target_pose(pose): 
    target_pose = geometry_msgs.PoseStamped()
    target_pose.header.frame_id = 'map'
    target_pose.header.stamp = rospy.Time.now()
    target_pose.pose.position.x = pose[0]
    target_pose.pose.position.y = pose[1]
    target_pose.pose.position.z = pose[2]
    target_pose.pose.orientation = euler_to_quaternion(pose[3])
    return target_pose

def euler_to_quaternion(orientation_z):
    radian = orientation_z * (PI / 180)
    q = quaternion_from_euler(0., 0., radian)
    return Quaternion(q[0], q[1], q[2], q[3])

if __name__=="__main__":
    main()
