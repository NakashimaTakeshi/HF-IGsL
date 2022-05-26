#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import print_function,unicode_literals, absolute_import
import sys
sys.path.append("../Serket/")

import serket as srk
import serket_ros as srkros
import mlda
import CNN
import gmm
import rospy
from serket.utils import Buffer

def main():
    rospy.init_node( "place_categorization" )

    obs1 = srkros.ObservationPos( "/odom" )
    obs1_buf = Buffer()
    gmm1 = gmm.GMM( 10 )

    obs2 = srkros.ObservationImg( "/camera/rgb/image_raw" )
    cnn1 = CNN.CNNFeatureExtractor( fileames=None )
    cnn1_buf = Buffer()

    mlda1 = mlda.MLDA( 10 )

    obs1_buf.connect( obs1 )
    gmm1.connect( obs1_buf )
    cnn1.connect( obs2 )
    cnn1_buf.connect( cnn1 )
    mlda1.connect( gmm1, cnn1_buf )

    n = 0
    while not rospy.is_shutdown():
        raw_input( "Hit enter to update the integrated model." )
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
        n += 1

if __name__=="__main__":
    main()
