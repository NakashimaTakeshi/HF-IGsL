#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals
from std_msgs.msg import String
import rospy
import pandas as pd

class WordPublisher ():
    def __init__(self, world):
        self.pub = rospy.Publisher('serket_ros/word_publisher/word', String, queue_size=10)
        self.df_cur_world_sentences = self.read_cur_world_sentences_from_csv_file(world)
        
    def read_cur_world_sentences_from_csv_file(self, cur_world):
        df = pd.read_csv("../sentences/aws_robomaker_world_sentences.csv")
        df_cur_world = df[df["world"] == cur_world]
        return df_cur_world

    def publish_word(self, roomIndex):
        place_word = self.df_cur_world_sentences[self.df_cur_world_sentences["index"] == roomIndex].sample(n=1)["sentences"].values[0]
        self.pub.publish(place_word)

    def publish_word_manual(self, place_word):
        self.pub.publish(place_word)
