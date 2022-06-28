#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import waypoint_navigation as waynavi
import image_feature
import threading


class SerketThread (threading.Thread):
    exitFlag = True

    def __init__(self, threadID, name):
        threading.Thread.__init__(self)
        self._stop_event = threading.Event()
        self.threadID = threadID
        self.name = name
        self.waypoint_navigation = waynavi.WaypointNavigation(world="aws_robomaker_small_house_world", each_area_point_number=10)
        self.imageFeature = image_feature.ImageFeature()
        self.poses = self.waypoint_navigation.poses
        self.counter = len(self.poses)

    def run(self):
        rospy.loginfo("Starting " + self.name +".")
        self.execute(self.name, self.counter)
        rospy.loginfo("Exiting " + self.name + ".")

    def stop(self):
        self._stop_event.set()

    def execute(self, threadName, counter):
        while True:
            if counter == 0:
                break
            if (threadName == "ImageFeature"):
                rospy.loginfo("Starting Serket categorization.")
                while SerketThread.exitFlag:
                    self.imageFeature.update()
                SerketThread.exitFlag = True
            if (threadName == "WaypointNavigation"):
                # Set waypoint navigation for one area
                pose = self.poses.pop(0)
                for i in range(len(pose)):
                    rospy.loginfo("Starting move, the goal pose is " + str(pose[i]) + ".")
                    self.waypoint_navigation.execute(pose[i])
                SerketThread.exitFlag = False
            counter -= 1

        self.stop()


def main():
    rospy.init_node( "serket_thread" )
    # Create new threads
    thread1 = SerketThread(1, "ImageFeature")
    thread2 = SerketThread(2, "WaypointNavigation")
    # Start new Threads
    thread1.start()
    thread2.start()


if __name__=="__main__":
    main()
