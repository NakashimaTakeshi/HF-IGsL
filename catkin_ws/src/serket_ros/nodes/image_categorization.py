#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import print_function,unicode_literals, absolute_import
import sys
sys.path.append("../Serket/")

import serket as srk
import serket_ros as srkros
import mlda
import CNN
import rospy

def main():
    rospy.init_node( "image_categorization" )

    obs1 = srkros.ObservationImg( "/camera/rgb/image_raw" )
    #obs1 = srkros.ObservationImg( "/serket_ros/image_publisher/image" )
    cnn1 = CNN.CNNFeatureExtractor( fileames=None )
    mlda1 = mlda.MLDA( 3, [1000] )

    cnn1.connect( obs1 )
    mlda1.connect( cnn1 )

    n = 0
    while not rospy.is_shutdown():
        raw_input( "Hit enter to update the integrated model." )
        obs1.update()
        print("'obs1' module updated.")
        cnn1.update()
        print("'cnn1' module updated.")
        mlda1.update()
        print("'mlda1' module updated.")
        n += 1

if __name__=="__main__":
    main()
