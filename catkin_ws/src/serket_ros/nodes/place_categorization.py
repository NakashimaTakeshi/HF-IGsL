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
from serket.utils import Buffer
import numpy as np
import matplotlib.pyplot as plt

def main():
    rospy.init_node( "place_categorization" )

    cat_nbr = 10

    obs1 = srkros.ObservationPos( "/odom" )
    obs1_buf = Buffer()
    gmm1 = gmm.GMM( cat_nbr )

    obs2 = srkros.ObservationImg( "/camera/rgb/image_raw" )
    cnn1 = CNN.CNNFeatureExtractor( fileames=None )
    cnn1_buf = Buffer()

    mlda1 = mlda.MLDA( cat_nbr )

    obs1_buf.connect( obs1 )
    gmm1.connect( obs1_buf )
    cnn1.connect( obs2 )
    cnn1_buf.connect( cnn1 )
    mlda1.connect( gmm1, cnn1_buf )

    n = 0
    
    fig = plt.figure("Serket-ROS")
    ax = fig.add_subplot(1, 1, 1)

    cat_bins = range(1, cat_nbr+1)

    while not rospy.is_shutdown():
        input( "Hit enter to update the integrated model." )

        obs1.update()
        obs1_buf.update()
        print("'obs1' module updated.")
        obs2.update()
        print("'obs2' module updated.")
        gmm1.update()
        print("'gmm1' module updated.")
        cnn1.update()
        cnn1_buf.update()
        print("'cnn1' module updated.")
        mlda1.update()
        print("'mlda1' module updated.")

        if n != 0:
            mlda1_cat = np.loadtxt("../nodes/module006_mlda/%03d/categories.txt"%n, unpack='False')
            # print(mlda1_cat)

            ax.hist(mlda1_cat, histtype='bar', bins=cat_bins)
            ax.set_xticks(cat_bins)
            ax.set_xticklabels(cat_bins)
            ax.set_yticks(range(1, n+2))
            ax.set_yticklabels(range(1, n+2))
            ax.set_xlabel("Categories")
            ax.set_ylabel("Observations")
            ax.set_title("MLDA")
            ax.legend()
            ax.grid()
            ax.autoscale(False)

            plt.draw()
            plt.pause(0.001)
            ax.cla()

        n += 1

if __name__=="__main__":
    main()
