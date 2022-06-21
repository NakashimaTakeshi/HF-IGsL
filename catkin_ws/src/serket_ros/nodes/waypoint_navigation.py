#! /usr/bin/env python

from asyncore import write
import csv
import rospy
import actionlib
import tf
import csv

#from nav_msgs.msg import Odometry
import std_msgs.msg
import math
import time
from geometry_msgs.msg import Twist,Quaternion
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import Odometry
# from rgiro_spco2_slam.srv import spco_speech


class ReplayScenario():
    def __init__(self, cmd_vel):
        self.pub = rospy.Publisher(cmd_vel, Twist, queue_size=1)
        # self.pub2 = rospy.Publisher('speech_to_text', std_msgs.msg.String, queue_size=10) ## queue size is not important for sending just one messeage.
        # self.srv = rospy.ServiceProxy('rgiro_spco2_slam/spcospeech',spco_speech)
        # self.client = actionlib.SimpleActionClient('move_base'vel MoveBaseAction) 
        # self.client.wait_for_server()
        self.state = "phase1"
 
    def reading_scenario(self, world): 
        self.scenario = []
        for line in open( '../waypoints/' + world + '.csv', 'r'):
            readitems = line[:-1].split(',')
            if not (readitems[0].startswith('#') or line==('\n')):
                self.scenario.append( readitems )

    def pub_cmd_vel(self,target_position_x,target_position_y,target_position_theta,linear_speed,angular_speed):
        ## Read the coordinates of the robot in the gazebo world.
        current_position = rospy.wait_for_message('tracker', Odometry)
        print(current_position)

        ## allowable pose error.
        error_distance = 0.05
        if self.state == "phases2":
            error_distance = error_distance * 3
        error_direction = 0.20
        error_orientation =0.10

        # Calculate delta_distance, delta_direction and  delta_orientation.
        current_orientation = tf.transformations.euler_from_quaternion((current_position.pose.pose.orientation.x,current_position.pose.pose.orientation.y,current_position.pose.pose.orientation.z,current_position.pose.pose.orientation.w))
        direction = math.atan2((target_position_y - current_position.pose.pose.position.y),(target_position_x - current_position.pose.pose.position.x))
        delta_direction = direction - current_orientation[2]
        delta_distance = math.sqrt((target_position_x - current_position.pose.pose.position.x)**2 +(target_position_y - current_position.pose.pose.position.y)**2 )
        #if (target_position_x - current_position.pose.pose.position.x) < 0:
        #    delta_direction = delta_direction - math.pi

        delta_orientation = math.radians(target_position_theta) - current_orientation[2]
        
        print('{} START'.format(self.state))
        print("----------------- Phase 1 -----------------")
        print("target_position(x,y):" + '{:.2f} {:.2f}'.format(target_position_x, target_position_y))
        print("current_position(x,y):" + '{:.2f} {:.2f}'.format(current_position.pose.pose.position.x, current_position.pose.pose.position.y))
        print("delta_distance" + '{:.2f}'.format(delta_distance) + " < " + error_distance + "m")
        print("direction:"+'{:.2f}'.format(direction)+"[rad.]"+'{:.2f}'.format(math.degrees(direction))+"[deg.]")
        print("current_orientation:"+'{:.2f}'.format(current_orientation[2])+"[rad.]"+'{:.2f}'.format(math.degrees(current_orientation[2]))+"[deg.]")
        print("delta_direction:"+'{:.2f}'.format(math.degrees(delta_direction))+"  <  "+'{:.2f}'.format(math.degrees(error_direction))+"[deg.]")
        print(" ")
        print("----------------- Phase 2 -----------------")
        print("target_orientation:"+'{:.2f}'.format(math.radians(target_position_theta))+"[rad.]"+target_position_theta+"[deg.]")
        print("current_orientation:"+'{:.2f}'.format(current_orientation[2])+"[rad.]"+'{:.2f}'.format(math.degrees(current_orientation[2]))+"[deg.]")
        print("delta_orientation:"+'{:.2f}'.format(math.degrees(delta_orientation))+"  <  "+'{:.2f}'.format(math.degrees(error_orientation))+"[deg.]")
        print(" ")


        if((delta_direction  % (2 * math.pi)) <= math.pi ):
            delta_direction = delta_direction  % (2 * math.pi)
        else:
            delta_direction = (delta_direction  % (2 * math.pi)) - 2 * math.pi
        print("delta_direction:"+'{:.2f}'.format(delta_direction)+"[rad.]"+'{:.2f}'.format(math.degrees(delta_direction))+"[deg.]")

        if(( delta_orientation  % (2 * math.pi)) <= math.pi ):
            delta_orientation = delta_orientation  % (2 * math.pi)
        else:
            delta_orientation = (delta_orientation  % (2 * math.pi)) - 2 * math.pi
        print("delta_orientation"+'{:.2f}'.format(delta_orientation)+"[rad.]"+'{:.2f}'.format(math.degrees(delta_orientation))+"[deg.]")

        # Calculate delta_distance, delta_direction and  delta_orientation.
        cmd_velocity = Twist()
        cmd_velocity.linear.x = math.sqrt(current_position.twist.twist.linear.x ** 2 + current_position.twist.twist.linear.y ** 2) *0.8
        # cmd_velocity.linear.y = current_position.twist.twist.linear.y
        
        if(delta_distance >= error_distance):
            if(abs(delta_direction) >= error_direction):
                cmd_velocity.angular.z = math.copysign(angular_speed,delta_direction) * math.pi / 180.0
            else:
                if(abs(delta_distance) <= error_distance * 30) :linear_speed = linear_speed * delta_distance / (error_distance * 30)
                # cmd_velocity.angular.z = 0
                cmd_velocity.angular.z =  angular_speed * delta_direction / error_direction * math.pi / 180.0
                print("delta_direction / error_direction "+delta_direction / error_direction)
                cmd_velocity.linear.x = linear_speed

            self.pub.publish(cmd_velocity)
            self.state = "phases1"
            print("cmd_velocity:Phase1"+cmd_velocity)

        else:
            if(abs(delta_orientation) >= error_orientation):
                cmd_velocity.angular.z = math.copysign(angular_speed,delta_orientation)  * math.pi / 180.0
                # cmd_velocity.angular.z = angular_speed * delta_orientation / error_orientation * math.pi / 180.0
            else:
                cmd_velocity.angular.z = 0
                cmd_velocity.linear.x = 0
                self.pub.publish(cmd_velocity)
                print("cmd_velocity:Phase3"+cmd_velocity)
                return True

            # if abs(delta_orientation) <= error_orientation * 5 :cmd_velocity.angular.z = cmd_velocity.angular.z / 5
            self.pub.publish(cmd_velocity)
            self.state = "phases2"
            print("cmd_velocity:Phase2"+cmd_velocity)
        
        return False



def print_current_location(base_link):
    now = rospy.Time.now()
    listener.waitForTransform("map", base_link, now, rospy.Duration(4.0))
    # Get the current position in map coordinate system from tf
    position, quaternion = listener.lookupTransform("map", base_link, now)
    print("base_link"+position+quaternion)
    current_position = rospy.wait_for_message('tracker', Odometry)
    print("position_in_gazebo"+current_position.pose.pose)

    # filepath="/root/TurtleBot3/catkin_ws/src/serket_ros/output"
    # with open(filepath,"a") as f:
    #     diff_distance = math.sqrt((position[0] - current_position.pose.pose.position.x)**2 +(position[1] - current_position.pose.pose.position.y)**2 )
    #     belief_orientation = tf.transformations.euler_from_quaternion(quaternion)
    #     real_orientation = tf.transformations.euler_from_quaternion([current_position.pose.pose.orientation.x,current_position.pose.pose.orientation.y,current_position.pose.pose.orientation.z,current_position.pose.pose.orientation.w])
    #     diff_orientation = real_orientation[2] - belief_orientation[2]
    #     # writer.writerow([now2,diff_distance,position,diff_orientation, quaternion,current_position.pose.pose])  
    #     writer = csv.writer(f)
    #     writer.writerow([now.secs,diff_distance,diff_orientation,current_position.pose.pose.position.x,current_position.pose.pose.position.y,real_orientation[2],position[0],position[1],belief_orientation[2]])  

def goal_pose(pose): 
    goal_pose = MoveBaseGoal()
    goal_pose.target_pose.header.frame_id = 'map'
    goal_pose.target_pose.pose.position.x = pose[0][0]
    goal_pose.target_pose.pose.position.y = pose[0][1]
    goal_pose.target_pose.pose.position.z = pose[0][2]
    goal_pose.target_pose.pose.orientation.x = pose[1][0]
    goal_pose.target_pose.pose.orientation.y = pose[1][1]
    goal_pose.target_pose.pose.orientation.z = pose[1][2]
    goal_pose.target_pose.pose.orientation.w = pose[1][3]

    return goal_pose

def wait_to_reaching(goal): 
    while True:
        now = rospy.Time.now()
        listener.waitForTransform("map", "tb3_1/base_link", now, rospy.Duration(4.0))

        # Get the current position in map coordinate system from tf
        position, quaternion = listener.lookupTransform("map", "tb3_1/base_link", now)
        # When a robot comes within 0.01 meter around and 0.01 radians(0.3degree) the waypoint goal, and issue the next waypoint.
        #print math.sqrt((position[0]-goal.target_pose.pose.position.x)**2 + (position[1]-goal.target_pose.pose.position.y)**2)
        #print 2 * ( math.atan(quaternion[2]/quaternion[3])-math.atan(goal.target_pose.pose.orientation.z/goal.target_pose.pose.orientation.w))
        if(math.sqrt((position[0]-goal.target_pose.pose.position.x)**2 + (position[1]-goal.target_pose.pose.position.y)**2 ) <= 0.05)\
            and(2*(math.atan(quaternion[2]/quaternion[3])-math.atan(goal.target_pose.pose.orientation.z/goal.target_pose.pose.orientation.w)) <= 0.1):
            print("reaching")
            rospy.sleep(2)
            break

        else:
            rospy.sleep(0.5)

    return True

if __name__ == '__main__':
    # receive args
    base_link = 'base_link'
    cmd_vel = 'cmd_vel'
    world = 'aws_robomaker_small_house_world'
    
    rospy.init_node('replay_scenario')
    # print "waiting"
    dammy_time = rospy.Time.now()#time.time()
    while dammy_time == rospy.Time.now():
        dammy_time = rospy.Time.now()
    # print "b"
    listener = tf.TransformListener()
    listener.waitForTransform("map", base_link, rospy.Time(), rospy.Duration(4.0))

    # print "c"
    play1 = ReplayScenario(cmd_vel)
    # print "d"    
    play1.reading_scenario(world)
    # print "e"
    print(play1.scenario)
    # print "f"
    rospy.sleep(2)
    calculate_time = 1
 
    for teaching in play1.scenario:
        if(teaching[0])=="teaching":
            for i in range(int(teaching[2])): 
                str_msg = std_msgs.msg.String(teaching[1])
                rospy.sleep(5)
                # rospy.wait_for_service('rgiro_spco2_slam/spcospeech')
                # try:
                #     # play1.srv = rospy.ServiceProxy('rgiro_spco2_slam/spcospeech',spco_speech)
                #     spcospeech_service_result = play1.srv(teaching[1])
                # except rospy.ServiceException as exc_srv_spcospeech:
                #     print("Service did not process request: " + str(exc_srv_spcospeech)) 
                # if spcospeech_service_result  == False:
                #     rospy.logerr("Services are failed")

                # play1.pub2.publish(str_msg)
                # print_current_location()
                print("next!")
                # play1.client.send_goal(goal)
                # rospy.sleep(1.5 * calculate_time + 2)
                # calculate_time = calculate_time + 1
                

        elif(teaching[0])=="move":
            while not(play1.pub_cmd_vel(float(teaching[1]),float(teaching[2]),float(teaching[3]),float(teaching[4]),float(teaching[5]))):
                rospy.sleep(0.00001)
            print("finish moving")
            print_current_location(base_link)

        else:
            print("no comand error")
            rospy.sleep(1)
            break
