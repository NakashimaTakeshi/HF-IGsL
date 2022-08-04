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
    def __init__(self, cat_nbr):

        # Initialize the plot configuration
        self.cat_nbr = cat_nbr
        self.fig = plt.figure("Place Categorization")
        self.ax = self.fig.add_subplot(1, 1, 1)

        # Initialize the gmm and mlda model
        self.gmm1 = gmm.GMM(cat_nbr)
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
        self.mlda1.connect(self.gmm1, self.cnn1_buf, self.bow1_buf)
        
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
    
    def data_visualization(self, vis_timer, world) :
        cat_bins = range(1, self.cat_nbr+1)

        mlda1_cat = np.loadtxt("../nodes/module001_mlda/%03d/categories.txt"%vis_timer, unpack='False')

        self.ax.hist(mlda1_cat, histtype='bar', bins=cat_bins)
        self.ax.set_xticks(cat_bins)
        self.ax.set_xticklabels(cat_bins)
        self.ax.set_yticks(range(1, vis_timer+2))
        self.ax.set_yticklabels(range(1, vis_timer+2))
        self.ax.set_xlabel("Categories")
        self.ax.set_ylabel("Observations")
        self.ax.set_title("Place categorization of " + world)
        self.ax.legend()
        self.ax.grid()
        self.ax.autoscale(False)

        plt.draw()
        plt.pause(0.001)
        self.ax.cla()

        rospy.loginfo("Draw the number " + str(vis_timer + 1) + " observation time plot")
        rospy.sleep(2)

def main():
    # Initialize the default parameters
    world = rospy.get_param('/world')
    isAttended = rospy.get_param('/is_attended')
    rospy.init_node( "place_categorization" )
    word_publisher = wordpub.WordPublisher(world=world)
    vis_timer = 0
    placeCategorization = PlaceCategorization(cat_nbr=10)

    if isAttended:
        # User will run place_categorization without waypoint navigation (Manual)
        while not rospy.is_shutdown():
            place_word = input("Please input a sentence to describe the current scene: ")
            word_publisher.publish_word_manual(place_word)
            input( "Hit enter to update the integrated model." )
            placeCategorization.update()
            if vis_timer == 0 or vis_timer == 1 :
                rospy.loginfo("The first time and the second time will not draw plot")
            else:
                placeCategorization.data_visualization(vis_timer, world)
            vis_timer += 1
            rospy.loginfo("-----------------------------------------------------------------------")
    else:
        # User will run place_categorization with waypoint navigation (Automatical)
        waypoint_navigation = waynavi.WaypointNavigation(world=world, each_area_point_number=10)
        for i in range(3):
            poses = waypoint_navigation.read_pose_from_csv_file(world=world, number=10)
            j = 0
            for pose in poses:
                for waypoint in pose:
                    rospy.loginfo("Starting move. ")
                    waypoint_navigation.execute(waypoint)
                    word_publisher.publish_word(j)
                    placeCategorization.update()
                    if vis_timer == 0 or vis_timer == 1 :
                        rospy.loginfo("The first time and the second time will not draw plot")
                    else:
                        placeCategorization.data_visualization(vis_timer, world)
                    vis_timer += 1
                    rospy.loginfo("-----------------------------------------------------------------------")
                j += 1

if __name__=="__main__":
    main()
