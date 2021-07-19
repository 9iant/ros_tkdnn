#!/usr/bin/env python
from __future__ import print_function

import roslib
#roslib.load_manifest('depth_sub')
import sys
import rospy
import cv2
import std_msgs
from std_msgs.msg import String
from std_msgs.msg import UInt16
from std_msgs.msg import UInt8

from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry

from ros_tkdnn.msg import yolo_coordinateArray
from ros_tkdnn.msg import depth_info, depth_infoArray
from tf.transformations import *
import numpy as np
import numpy.matlib

# import matplotlib.pyplot as plt

import tf
import turtlesim.msg

from visualization_msgs.msg import Marker, MarkerArray
import pandas as pd


#-----------------from SORT------------------
import os
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import glob
import time
import argparse
from filterpy.kalman import KalmanFilter
#---------------------------------------------

np.random.seed(0)
colors = np.array([[255,0,0],[128,0,0],[255,255,0],[128,128,0],[0,255,0],[0,0,255]],dtype=float)/255.0
colors = np.matlib.repmat(colors,20,1)

class mot:
  def __init__(self):
  	self.depth_sub = rospy.Subscriber("/depth_estimator/depth_infoarray", depth_infoArray, self.detection_cb)
  	# publisher or srv server 
  	self.distance_threshold = 1.0 # [m]
  	self.mot_tracker = Sort(dist_threshold=self.distance_threshold) ## 
	self.mot_Array_pub = rospy.Publisher("/mot/tracks", MarkerArray, queue_size = 1)

  def publish_markerArray(self):
    if len(self.mot_tracker.trackers)==0: # if there is no tracks
	  print("there is no tracks!")
	  pass

    else:
      to_pub = MarkerArray()
      for tracker in self.mot_tracker.trackers:
        a_marker = Marker()
        a_marker.header.frame_id = "/map"
        a_marker.header.stamp = rospy.Time.now()
        a_marker.ns = "my_namespace"
        a_marker.id = tracker.id
        a_marker.type = Marker.SPHERE
        a_marker.action = Marker.ADD

        a_marker.pose.position.x = tracker.position[0]
        a_marker.pose.position.y = tracker.position[1]
        a_marker.pose.position.z = 1.0
        a_marker.pose.orientation.x = 0.0
        a_marker.pose.orientation.y = 0.0
        a_marker.pose.orientation.z = 0.0
        a_marker.pose.orientation.w = 1.0
        a_marker.scale.x = 0.5
        a_marker.scale.y = 0.5
        a_marker.scale.z = 0.5
        a_marker.color.a = 0.8 
        a_marker.color.r = colors[tracker.id][0]
        a_marker.color.g = colors[tracker.id][0]
        a_marker.color.b = colors[tracker.id][0]
        to_pub.markers.append(a_marker)

      self.mot_Array_pub.publish(to_pub)
      print("track published, current {} tracks".format(len(self.mot_tracker.trackers)))  



  def detection_cb(self, data):
  	# TODO: 
  	if len(data.results)>0:
  	  # only dog now.
  	  only_dog_indice = []
  	  for idx in range(len(data.results)):
	  	if data.results[idx].label == "dog":
	  	  only_dog_indice.append(idx)
	  if len(only_dog_indice):
	  	print("{} dogs are detected! ".format(len(only_dog_indice)))
	  	dets = np.zeros((len(only_dog_indice), 2))
	  	for i in only_dog_indice:
	  		dets[i][0] = data.results[i].x
	  		dets[i][1] = data.results[i].y
	  	self.mot_tracker.update(dets)##
	self.publish_markerArray()


  	# if data.label == "dog" :
  	# 	self.depth_result["dog"].append([data.x, data.y, data.confidence])
  	# 	#print("i get a dog!")
  	# elif data.label == "soldier":
  	# 	self.depth_result["soldier"].append([data.x, data.y, data.confidence])
  	# 	#print("i get a human!")
  	# print("dog : {}".format(len(self.depth_result["dog"])))
  	# print("soldier : {}".format(len(self.depth_result["soldier"])))


class Sort(object):
  def __init__(self, dist_threshold=0.3):
    """
    Sets key parameters for SORT
    """
    # self.max_age = max_age
    # self.min_hits = min_hits
    self.dist_threshold = dist_threshold
    self.trackers = [] ## important
    self.frame_count = 0

  def update(self, dets):
    """
    Params:
      dets - a numpy array of detections in the format [[x,y,score],[x,y,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.
    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    # get predicted locations from existing trackers.

    trks = np.zeros((len(self.trackers), 2))
    for idx, tracker in enumerate(self.trackers):
    	trks[idx] = self.trackers[idx].position # no predition

    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(np.array(dets), np.array(trks), self.dist_threshold) ###########3


    # update matched trackers with assigned detections
    for m in matched:
      print("matching : {}".format(m)) #################### m[0] / m[1] exchaged!
      print(dets[m[1], :])
      print(self.trackers[m[0]].position)
      self.trackers[m[0]].update(dets[m[1], :])

    # create and initialise new trackers for unmatched detections

    for i in unmatched_dets:
        trk = Tracker(dets[i,:])
        self.trackers.append(trk)
        print("new Track generated! current total {} tracks".format(Tracker.count))
    i = len(self.trackers)

    # if self.frame_count%150==1: # per 100 frames, delete tracker with few detections 
    #   del_trk = []
    #   for idx, tracker in enumerate(self.trackers):
    #     if tracker.hits < 10:
    #       del_trk.append(idx)
    #   for i in del_trk:
    #   	self.trackers.pop(i) # remove tracker with few detections
    #   	print("track dead because few detections")


    # for trk in reversed(self.trackers):
    #     d = trk.get_state()[0]
    #     if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
    #       ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
    #     i -= 1
    #     # remove dead tracklet
    #     if(trk.time_since_update > self.max_age):
    #       self.trackers.pop(i)
    # if(len(ret)>0):
    #   return np.concatenate(ret)
    # return np.empty((0,5))


def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))

def get_distancemat(dets, trks):
   """
   make distance matrix 
   """
   dets = np.expand_dims(dets,0)
   trks = np.expand_dims(trks,1)

   distance = np.linalg.norm(dets - trks, axis = 2)
   return distance

class Tracker(object):
  count = 0
  def __init__(self, position):
    self.id = Tracker.count
    self.position = np.array(position) # [x, y]
    Tracker.count += 1
    self.hits = 1
  def update(self, position): # just simple moving avg
    self.hits += 1
    self.position = (self.position*(self.hits-1) + np.array(position))/self.hits
    
def associate_detections_to_trackers(detections,trackers,distance_threshold = 1.0):
  """
  Assigns detections to tracked object (both represented as bounding boxes)
  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """

  if(len(trackers)==0):
    print("There is no trackers")
    return np.zeros((0,2),dtype=int), np.arange(len(detections)), np.zeros((0,2),dtype=int)

  distance_matrix = get_distancemat(detections, trackers)

  distance_matrix = np.where(distance_matrix>distance_threshold, 999, distance_matrix) # like gating
  if min(distance_matrix.shape) > 0:
      matched_indices = linear_assignment(distance_matrix)
  else:
    matched_indices = np.zeros((0,2))

  unmatched_detections = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with far distance
  matches = []
  for m in matched_indices:
    if(distance_matrix[m[0], m[1]] > distance_threshold): # if distance is too far, don't match
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def main(args):

  rospy.init_node('mot', anonymous=True)
  ic = mot()
 
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
