#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append("../Serket/")

import serket as srk
import serket_ros as srkros
import mlda
import CNN
import gmm
import rospy
import bow
from serket.utils import Buffer
import numpy as np
import matplotlib.pyplot as plt
import waypoint_navigation as waynavi
import word_publisher as wordpub

class PlaceCategorization ():
    def __init__(self):

        # Initialize the gmm and mlda model
        cat_nbr = 10
        self.gmm1 = gmm.GMM( cat_nbr )
        self.mlda1 = mlda.MLDA(cat_nbr)

        # Robot self-pose information
        self.obs1 = srkros.ObservationPos( "/odom" )
        self.obs1_buf = Buffer()
        # Connect self-pose to gmm model
        self.obs1_buf.connect(self.obs1)
        self.gmm1.connect(self.obs1_buf)

        # Image feature
        self.obs2 = srkros.ObservationImg( "/camera/rgb/image_raw" )
        self.cnn1 = CNN.CNNFeatureExtractor( fileames=None )
        self.cnn1_buf = Buffer()
        # Connect image feature to cnn model
        self.cnn1.connect(self.obs2)
        self.cnn1_buf.connect(self.cnn1)

        # Sentence
        self.obs3 = srkros.ObservationString("/serket_ros/word_publisher/word")
        self.bow1 = bow.BoW()
        self.bow1_buf = Buffer()
        # Connect sentence to Bow model
        self.bow1.connect(self.obs3)
        self.bow1_buf.connect(self.bow1)

        # Combine gmm and cnn model into mlda model
        self.mlda1.connect(self.gmm1, self.cnn1_buf)
        
    def update(self):
        rospy.loginfo("Starting categorization updated process.")
        self.obs1.update()
        self.obs1_buf.update()
        rospy.loginfo("'obs1' module updated.")

        self.obs2.update()
        rospy.loginfo("'obs2' module updated.")

        self.obs3.update()
        rospy.loginfo("'obs3' module updated.")

        self.gmm1.update()
        rospy.loginfo("'gmm1' module updated.")

        self.cnn1.update()
        self.cnn1_buf.update()
        rospy.loginfo("'cnn1' module updated.")

        self.bow1.update()
        self.bow1_buf.update()
        rospy.loginfo("'bow1' module updated.")

        self.mlda1.update()
        rospy.loginfo("'mlda1' module updated.")
        rospy.loginfo("Categorization updated successfully.")

def main():
    rospy.init_node( "place_categorization" )
    waypoint_navigation = waynavi.WaypointNavigation(world="aws_robomaker_bookstore_world", each_area_point_number=10)
    word_publisher = wordpub.WordPublisher(world="aws_robomaker_bookstore_world")
    placeCategorization = PlaceCategorization()
    for i in range(3):
        poses = waypoint_navigation.read_pose_from_csv_file(world="aws_robomaker_bookstore_world", number=10)
        j = 0
        for pose in poses:
            for waypoint in pose:
                rospy.loginfo("Starting move. ")
                waypoint_navigation.execute(waypoint)
                word_publisher.publish_word(j)
                placeCategorization.update()
                rospy.loginfo("-----------------------------------------------------------------------")
            j += 1
    # for pose in poses:
    #     for waypoint in pose:
    #         # rospy.loginfo("Starting move. current pose " + str(waypoint) + ".")
    #         rospy.loginfo("Starting move. ")
    #         waypoint_navigation.execute(waypoint)
    #         placeCategorization.update()
    #         rospy.loginfo("-----------------------------------------------------------------------")
    

if __name__=="__main__":
    main()
