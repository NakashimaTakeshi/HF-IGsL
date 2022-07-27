#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

def main():
    rospy.init_node('image_publisher', anonymous=True)

    pub = rospy.Publisher('serket_ros/image_publisher/image', Image, queue_size=10)
    bridge = CvBridge()


    for i in range(6):
        img = cv2.imread( "../Serket/examples/ROS/images/%03d.png" % i )
        msg = bridge.cv2_to_imgmsg(img, encoding=str("bgr8"))


        input( "Hit enter to publish." )
        pub.publish( msg )
        print( "Published." )

if __name__ == '__main__':
    main()
