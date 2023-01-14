#!/usr/bin/env python3
# coding: utf-8
from ros_rssm.srv import *
import rospy

i = 1

def handle_add_two_ints(req):
    global i
    print("i="+str(i))
    i += 1
    print("Returning [%s + %s = %s]"%(req.a, req.b, (req.a + req.b)))
    return AddTwoIntsResponse(req.a + req.b)
def add_two_ints_server():
    
    rospy.init_node('add_two_ints_server')
    s = rospy.Service('add_two_ints', AddTwoInts, handle_add_two_ints)
    print("Ready to add two ints.")
    
    rospy.spin()
if __name__ == "__main__":
    i = 1
    print("aaaaaaaaaaaaa")
    add_two_ints_server()