#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals
from std_msgs.msg import String
import rospy
import random

class WordPublisher ():
    def __init__(self, world):
        # Initialize the room dict
        self.kitchenDict = {'1': 'i heard something crashing in the kitchen in the middle of the night', '2': 'mother is in the kitchen',
         '3': 'every morning she helps her mother to prepare breakfast in the kitchen', '4': 'there is a cooking table next to the refrigerator',
         '5': 'they are talking in the kitchen', '6': 'we were cooking tempura at that time', 
         '7': 'mother hummed to herself as she went about her cooking', '8': 'she finished up lunch with coffee',
         '9': 'opening the refrigerator, I noticed the meat had spoiled', '10': 'nothing remained in the refrigerator'}

        self.livingRoomDict = {'1': 'the living room adjoins the dining room', '2': 'the living room in my new house is very large',
         '3': 'father made our living room more spacious', '4': 'one Sunday morning George burst into the living room and said the following',
         '5': 'our living room is sunny', '6': 'she put the blanket over the child sleeping on the sofa',
         '7': 'as he was tired, he was lying on the sofa with his eyes closed', '8': 'i am tired of watching television',
         '9': 'my father usually watches television after dinner', '10': 'there is a dictionary on the desk'}

        self.bedroomDict = {'1': 'i went up to my bedroom on tiptoe', '2': 'i have my own bedroom at home',
         '3': 'i found myself lying in my bedroom', '4': 'i share a bedroom with my sister',
         '5': 'i like to have a full-length mirror in my bedroom', '6': 'to be a good child, you need to go to bed and get up early', 
         '7': 'i go to bed early at night', '8': "she put on her sister's jeans and looked in the mirror",
         '9': 'a good conscience is a soft pillow', '10': 'the alarm clock is ten minutes fast'}

        self.counterDict = {'1': 'please pay at this counter', '2': 'is there room at the counter',
         '3': 'where is the information counter', '4': 'please come to the counter at least an hour before your flight',
         '5': 'you can get it at a bookstore', '6': 'he bought a number of books at the bookstore',
         '7': 'he bought an English book at a bookstore', '8': 'have you finished reading the book I lent you the other day', 
         '9': 'i want you to return the book I lent you the other day', '10': 'did you pay for the book'}
        
        self.bookshelfDict = {'1': 'he has a large number of books on his bookshelf', '2': 'he fixed the bookshelf to the wall', 
         '3': 'he made her a bookshelf', '4': 'i have a large number of books on my bookshelf', 
         '5': 'our living room is sunny', '6': 'the bookshelf is built in', 
         '7': 'carry these books back to the bookshelf', '8': 'my son has read every book on that shelf', 
         '9': 'the other day I came across a book that you might like', '10': 'the teacher asked me which book I liked'}

        self.restAreaDict = {'1': 'they had a rest for a while', '2': 'we took a rest for a while', 
         '3': 'i want to rest a little because all the homework is finished', '4': 'i share a bedroom with my sister', 
         '5': 'i felt better after I took a rest', '6': 'my grandfather would often read and study at this desk', 
         '7': 'i want to take a rest', '8': 'may I share this table with you', 
         '9': 'it makes me feel cheerful', '10': 'some people relax by reading'}

        self.pub = rospy.Publisher('serket_ros/word_publisher/word', String, queue_size=10)

        self.world = world
        

    def publish_word(self, roomIndex):
        wordIndex = random.randint(1,10)
        place_word = 'here is nothing'
        if self.world == "aws_robomaker_small_house_world":
            if roomIndex == 0:
                place_word = self.kitchenDict[str(wordIndex)]
            elif roomIndex == 1:
                place_word = self.livingRoomDict[str(wordIndex)]
            elif roomIndex == 2:
                place_word = self.bedroomDict[str(wordIndex)]
        elif self.world == "aws_robomaker_bookstore_world":
            if roomIndex == 0:
                place_word = self.counterDict[str(wordIndex)]
            elif roomIndex == 1:
                place_word = self.bookshelfDict[str(wordIndex)]
            elif roomIndex == 2:
                place_word = self.restAreaDict[str(wordIndex)]
        self.pub.publish(place_word)
