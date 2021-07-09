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
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from ros_tkdnn.msg import yolo_coordinateArray
from ros_tkdnn.msg import depth_info
from tf.transformations import *
import numpy as np


# import matplotlib.pyplot as plt

import tf
import turtlesim.msg

from visualization_msgs.msg import Marker
import pandas as pd

class image_converter:

  def __init__(self):


    self.data = pd.DataFrame(columns=['label','x','y'])

    self.vis_pub = rospy.Publisher("/depth_estimator/vis",Marker, queue_size=1)
    self.depth_info_pub = rospy.Publisher("/depth_estimator/depth_info", depth_info, queue_size = 1)
    self.depth_info = depth_info()

    self.object_estimator = rospy.Publisher("/depth_estimator/object_position", PoseStamped, queue_size = 1)

    self.bridge = CvBridge()
    self.depth_sub = rospy.Subscriber("/d435/depth/image_raw",Image,self.depth_cb)
    self.yolo_sub = rospy.Subscriber("/yolo_output",yolo_coordinateArray,self.yolo_cb)

    

    
    
    self.listener = tf.TransformListener()

    self.slam_odom = rospy.Subscriber("/gazebo_camera_pose", PoseStamped, self.slam_cb)
   

    self.depth_flag = False
    self.slam_flag = False



  def slam_cb(self,data):
    self.slam_flag = True
    
    self.drone_x = data.pose.position.x
    self.drone_y = data.pose.position.y

    orientation_q = data.pose.orientation
    orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
    (self.roll, self.pitch, self.yaw) = euler_from_quaternion(orientation_list)



    

  def depth_cb(self,data):
    
    self.depth_flag = True
    self.cv_depth_image = self.bridge.imgmsg_to_cv2(data, data.encoding)

  def yolo_cb(self,data):
    # print(len(data.results),self.depth_flag, self.slam_flag)
    if data.results and (self.depth_flag == True) and (self.slam_flag == True):

      for idx in range(len(data.results)):

        # Get center of box
        self.center_of_box = (data.results[idx].x_center,data.results[idx].y_center)

        # Gaussian sampling 
        num_samples = 1000
        
        self.gaussian_sampling = np.random.multivariate_normal(self.center_of_box,
          [[data.results[idx].w,0],[0,data.results[idx].h]], num_samples)

        # Get depth each point

        self.depth_value_list = []

        for point in self.gaussian_sampling:

          self.depth_value_list.append(self.get_depth(int(point[0]), int(point[1])))  

        # Choose representative point (minimum depth)
      
        depth_min = np.max(self.depth_value_list)

        for depth in self.depth_value_list:
          
          if depth > 0 and depth < depth_min:

            depth_min = depth 

            if depth_min != 0:

              self.depth_value_gaussian = depth_min 

              object_x, object_y = self.get_object_pos(data.results[idx].x_center,data.results[idx].y_center,self.depth_value_gaussian)

              self.depth_info.x = object_x
              self.depth_info.y = object_y
              self.depth_info.label = data.results[idx].label
              self.depth_info_pub.publish(self.depth_info)

              import os

              os.system("echo {},{},{} >> pos.txt".format(data.results[idx].label, object_x.item(), object_y.item()))















          # if depth_min == 0:

          #   rospy.logwarn("minimum depth is 0, ERROR")
          

    
    

        # Get Obstacle Position based on SLAM odom

       # obstacle_x, obstacle_y = self.get_obstacle_pos(self.depth_value_gaussian, self.drone_x, self.drone_y, self.roll , self.pitch , self.yaw)
        # object_x, object_y = self.get_object_pos(data.results[idx].x_center,data.results[idx].y_center,self.depth_value_gaussian)
        # self.depth_info.depth = self.depth_value_gaussian/10.0 # centi meter

        # self.depth_info.label = data.results[idx].label

        # self.depth_info.x_center = int(object_x)

        # self.depth_info.y_center = int(object_y)
        # print(self.depth_info)
        # self.depth_estimator_pub.publish(self.depth_info)


        # self.draw_top_view(object_x,object_y,255,255,255)
        # self.draw_top_view(self.drone_x*100,self.drone_y*100,0,255,255)

  def get_object_pos(self,u,v,z):
    side = 'left'
    if side == 'left':
      fx = 382.3295018608584
      fy = 381.7717111167475
      cx = 320.9189895208528
      cy = 234.40496812669917

    elif side == 'right':
      fx = 382.28892862203827
      fy = 381.7841853580208
      cx = 320.69780805182154
      cy = 234.32116074974246


    # !! x,y,z's units are different !!
    #Pc
    x,y,z = z*np.linalg.inv(np.matrix([[fx,0,cx],[0,fy,cy],[0,0,1]])) * np.matrix([[u],[v],[1]])
   
    try:
      (trans, rot) = self.listener.lookupTransform('/world', 'camera_d435', rospy.Time(0))
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
      print("except")

    # print("Trans : ",trans)
    # print("ROt: ",rot)


    
    # get camera static tf and certify 
    # get transformation matrix Tmc 
    
    world_cam_T = tf.transformations.translation_matrix((trans[0],trans[1],trans[2]))
    world_cam_R = tf.transformations.quaternion_matrix((rot[0],rot[1],rot[2],rot[3]))
    # generate TF matrix in python(Tmc)

    Tmc = np.matmul(world_cam_T, world_cam_R)

   # print(Tmc)
    # Pm = Tmc * Pc ( Pc = [x y z 1]')
    
    Pc = np.array([[z.item()], [x.item()], [y.item()], [1]])
   # print(np.array([x.item(), y.item(), z.item(), 1]))
    #print(Pc)
    Pm = np.matmul(Tmc, Pc)
    # print("===")
    # print(Pm)
    # geometry_msgs/Posestamped

    Pm_pub = PoseStamped()

    Pm_pub.header.stamp = rospy.Time.now()
    Pm_pub.pose.position.x = Pm[0] / 1000.0 # mm -> m 
    Pm_pub.pose.position.y = Pm[1] / 1000.0 # mm -> m 
    Pm_pub.pose.position.z = Pm[2] / 1000.0 # mm -> m 
    
    self.draw_in_rviz(Pm_pub.pose.position.x,Pm_pub.pose.position.y, Pm_pub.pose.position.z)

    
    # self.object_estimator.publish(Pm)
    return Pm_pub.pose.position.x, Pm_pub.pose.position.y

  def draw_in_rviz(self,x,y,z):
    self.marker = Marker()

    self.marker.header.frame_id = "world"
    self.marker.header.stamp = rospy.Time.now()
    self.marker.ns = "my_namespace"
    self.marker.id = 0
    self.marker.type = 1
    self.marker.action = Marker.ADD
    self.marker.pose.position.x = x
    self.marker.pose.position.y = y
    self.marker.pose.position.z = z
    self.marker.pose.orientation.x = 0.0
    self.marker.pose.orientation.y = 0.0
    self.marker.pose.orientation.z = 0.0
    self.marker.pose.orientation.w = 1.0
    self.marker.scale.x = 0.5
    self.marker.scale.y = 0.5
    self.marker.scale.z = 0.5
    self.marker.color.a = 1.0 # Don't forget to set the alpha!
    self.marker.color.r = 0.0
    self.marker.color.g = 1.0
    self.marker.color.b = 0.0

    # self.marker.mesh_resource = "package://pr2_description/meshes/base_v0/base.dae"
    self.vis_pub.publish(self.marker)
    rospy.loginfo("marker has been published")
  def draw_top_view(self, x, y,r,g,b):

    
    x += 250
    y += 250

    img = cv2.circle(self.img, (int(x),int(y)), 1 ,(r,g,b), -1)

    cv2.imshow('img',img)
    cv2.waitKey(1)

    

    


    
            
  # def get_obstacle_pos (self,depth_gaussian,drone_x,drone_y,roll,pitch,yaw):
    
   
  #   depth_z = depth_gaussian/10 * np.cos(pitch)

  #   drone_x *= 100
  #   drone_y *= 100

  #   obstacle_x = drone_x + depth_z * np.cos(yaw)
    
  #   obstacle_y = drone_y + depth_z * np.sin(yaw)

  #   return obstacle_x, obstacle_y
    
  def get_depth(self,x,y):

    return self.cv_depth_image[x][y]
    


  def rgb_cb(self,data):
    try:
      #(480,640,3)

      self.cv_rgb_image = self.bridge.imgmsg_to_cv2(data,'bgr8')
      self.rgb_input = True
    except CvBridgeError as e:
      self.rgb_input = False
      rospy.logerr(e)



def main(args):
  rospy.init_node('depth_subscriber', anonymous=True)
  ic = image_converter()
 
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
