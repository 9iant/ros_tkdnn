"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import glob
import time
import argparse
from filterpy.kalman import KalmanFilter

np.random.seed(0)


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


# class Tracker(object):
#   """
#   This class represents the internal state of individual tracked objects observed as bbox.
#   """
#   count = 0
#   def __init__(self,position):
#     """
#     Initialises a tracker using initial position
#     """
#     self.position = position

#     self.time_since_update = 0 
#     self.id = KalmanBoxTracker.count ######################
#     KalmanBoxTracker.count += 1 #########################3
#     self.history = []
#     self.hits = 0
#     self.hit_streak = 0
#     self.age = 0

#   def update(self,bbox):
#     """
#     Updates the state vector with observed bbox.
#     """
#     self.time_since_update = 0
#     self.history = []
#     self.hits += 1
#     self.hit_streak += 1
#     self.kf.update(convert_bbox_to_z(bbox))



# def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
#   """
#   Assigns detections to tracked object (both represented as bounding boxes)
#   Returns 3 lists of matches, unmatched_detections and unmatched_trackers
#   """
#   if(len(trackers)==0): # if there is no tracker
#     return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int) # [ empty matches/ all detection---> unmatched ] | empty trackers.

#   iou_matrix = iou_batch(detections, trackers)

#   if min(iou_matrix.shape) > 0:
#     a = (iou_matrix > iou_threshold).astype(np.int32)
#     if a.sum(1).max() == 1 and a.sum(0).max() == 1:
#         matched_indices = np.stack(np.where(a), axis=1)
#     else:
#       matched_indices = linear_assignment(-iou_matrix)
#   else:
#     matched_indices = np.empty(shape=(0,2))

#   unmatched_detections = []
#   for d, det in enumerate(detections):
#     if(d not in matched_indices[:,0]):
#       unmatched_detections.append(d)
#   unmatched_trackers = []
#   for t, trk in enumerate(trackers):
#     if(t not in matched_indices[:,1]):
#       unmatched_trackers.append(t)

#   #filter out matched with low IOU
#   matches = []
#   for m in matched_indices:
#     if(iou_matrix[m[0], m[1]]<iou_threshold):
#       unmatched_detections.append(m[0])
#       unmatched_trackers.append(m[1])
#     else:
#       matches.append(m.reshape(1,2))
#   if(len(matches)==0):
#     matches = np.empty((0,2),dtype=int)
#   else:
#     matches = np.concatenate(matches,axis=0)

#   return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


# class Sort(object):
#   def __init__(self, max_age=100, min_hits=3, dist_threshold=0.3):
#     """
#     Sets key parameters for SORT
#     """
#     self.max_age = max_age
#     self.min_hits = min_hits
#     self.dist_threshold = dist_threshold
#     self.trackers = [] ## important
#     self.frame_count = 0

#   def update(self, dets=np.empty((0, 3))):
#     """
#     Params:
#       dets - a numpy array of detections in the format [[x,y,score],[x,y,score],...]
#     Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
#     Returns the a similar array, where the last column is the object ID.
#     NOTE: The number of objects returned may differ from the number of detections provided.
#     """
#     self.frame_count += 1
#     # get predicted locations from existing trackers.

#     trks = np.zeros((len(self.trackers), 3))
#     to_del = []
#     ret = []
#     """
#     for t, trk in enumerate(trks): # trk is a single row of trks 
#       pos = self.trackers[t].predict()[0]
#       trk[:] = [pos[0], pos[1], pos[2], pos[3], 0] # substituted!

#       if np.any(np.isnan(pos)):
#         to_del.append(t)"""

#     trks = np.ma.compress_rows(np.ma.masked_invalid(trks)) # remove invalid trks(predicted)

#     for t in reversed(to_del):
#       self.trackers.pop(t) # remove tracker with invalid pos!

#     matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.dist_threshold) ###########3


#     # update matched trackers with assigned detections
#     for m in matched:
#       self.trackers[m[1]].update(dets[m[0], :])

#     # create and initialise new trackers for unmatched detections
#     for i in unmatched_dets:
#         trk = KalmanBoxTracker(dets[i,:])
#         self.trackers.append(trk)
#     i = len(self.trackers)
#     for trk in reversed(self.trackers):
#         d = trk.get_state()[0]
#         if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
#           ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
#         i -= 1
#         # remove dead tracklet
#         if(trk.time_since_update > self.max_age):
#           self.trackers.pop(i)
#     if(len(ret)>0):
#       return np.concatenate(ret)
#     return np.empty((0,5))



if __name__ == '__main__':
  print("main")
  dets = [[0,0],[100,0],[1,0]]
  trks = [[0,0],[1,0],[10,0]]
  
  distance_threshold = 10

  dist = get_distancemat(dets,trks)
  print('distance matrix')
  print(dist)
  # dist[dist>distance_threshold] = np.Inf
  print("mask high distance")
  print(np.where(dist>distance_threshold, float('inf'), dist))
  dist = np.where(dist>distance_threshold, 999, dist)
  print("result")
  print(linear_assignment(dist))

  print("-"*20)

  a = (dist < distance_threshold).astype(np.float32)

  print(a)
  print(a.sum(0).max()==1 and a.sum(1).max() ==1)
  # if min(distance_matrix.shape) > 0:
  #   a = (distance_matrix > distance_threshold).astype(np.float32)
  #   if a.sum(1).max() == 1 and a.sum(0).max() == 1:
  #       matched_indices = np.stack(np.where(a), axis=1)
  #   else:
  #     matched_indices = linear_assignment(-iou_matrix)
  # else:
  #   matched_indices = np.empty(shape=(0,2))

  # unmatched_detections = []
  # for d, det in enumerate(detections):
  #   if(d not in matched_indices[:,0]):
  #     unmatched_detections.append(d)
  # unmatched_trackers = []
  # for t, trk in enumerate(trackers):
  #   if(t not in matched_indices[:,1]):
  #     unmatched_trackers.append(t)

  # mot_tracker = Sort()
  # # get detectes!
  # mot_tracker.update(dets)
  
