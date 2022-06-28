#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import print_function,unicode_literals, absolute_import
import sys
from tokenize import String
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

class ImageFeature ():
    def __init__(self):

        # Set waypoint navigation
        self.obs1 = srkros.ObservationString("serket_ros/move_trigger/trigger")
        # Initialize the gmm and mlda model
        self.cat_nbr = 10
        self.mlda1 = mlda.MLDA( self.cat_nbr )

        # Image feature
        self.obs2 = srkros.ObservationImg( "/camera/rgb/image_raw" )
        self.cnn1 = CNN.CNNFeatureExtractor( fileames=None )
        self.cnn1_buf = Buffer()
        # Connect image feature to cnn model
        self.cnn1.connect(self.obs2)
        self.cnn1_buf.connect(self.cnn1)

        # Combine cnn model into mlda model
        self.mlda1.connect(self.cnn1_buf)
    
        self.fig = plt.figure("image_feature")
        self.ax = self.fig.add_subplot(1, 1, 1)

        self.cat_bins = range(1, self.cat_nbr+1)
        self.update_argument = (self.obs2, self.cnn1, self.cnn1_buf, self.mlda1, self.ax, self.cat_bins)
        
    def update(self):
        self.update_argument[0].update()
        rospy.loginfo("'obs2' module updated.")

        self.update_argument[1].update()
        self.update_argument[2].update()
        rospy.loginfo("'cnn1' module updated.")

        self.update_argument[3].update()
        rospy.loginfo("'mlda1' module updated.")
