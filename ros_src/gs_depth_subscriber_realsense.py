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
from ros_tkdnn.msg import depth_info
from tf.transformations import *
import numpy as np


depth_width, depth_height = 1280, 720 ########################################
rgb_width, rgb_height = 640, 480 ################################################

width_factor = 1.0 * depth_width / rgb_width  
height_factor = 1.0 * depth_height / rgb_height

# import matplotlib.pyplot as plt

import tf
import turtlesim.msg

from visualization_msgs.msg import Marker
import pandas as pd

class image_converter:

  def __init__(self, depth_camera_topic, depth_camera_info_topic, yolo_output_topic, rgb_camera_info_topic, odometry_topic):


    self.data = pd.DataFrame(columns=['label','x','y'])

    self.vis_pub = rospy.Publisher("/depth_estimator/vis",Marker, queue_size=1)
    self.depth_info_pub = rospy.Publisher("/depth_estimator/depth_info", depth_info, queue_size = 1)
    self.depth_info = depth_info()

    self.object_estimator = rospy.Publisher("/depth_estimator/object_position", PoseStamped, queue_size = 1)

    self.bridge = CvBridge()

    self.depth_sub = rospy.Subscriber(depth_camera_topic,Image,self.depth_cb)
    self.yolo_sub = rospy.Subscriber(yolo_output_topic,yolo_coordinateArray,self.gs_yolo_cb)

    self.depth_camera_info_sub = rospy.Subscriber(depth_camera_info_topic,CameraInfo, self.depth_camera_info_cb)
    self.depth_intrinsics = None
    self.rgb_camera_info_sub = rospy.Subscriber(rgb_camera_info_topic,CameraInfo, self.rgb_camera_info_cb)
    self.rgb_intrinsics = None

    
    
    self.listener = tf.TransformListener()

    self.slam_odom = rospy.Subscriber(odometry_topic, Odometry, self.slam_cb) ##
   

    self.depth_flag = False
    self.slam_flag = False

  def depth_camera_info_cb(self, cameraInfo):
    try:
      if self.depth_intrinsics:
        return
      self.depth_intrinsics = {"width":0, "height":0,"fx":0,"fy":0,"cx":0,"cy":0}
      self.depth_intrinsics['width'] = cameraInfo.width
      self.depth_intrinsics['height'] = cameraInfo.height
      self.depth_intrinsics['fx'] = cameraInfo.K[0]
      self.depth_intrinsics['fy'] = cameraInfo.K[4]
      self.depth_intrinsics['cx'] = cameraInfo.K[2]
      self.depth_intrinsics['cy'] = cameraInfo.K[5]
      print(self.depth_intrinsics)
  
    except CvBridgeError as e:
        print(e)
        return

  def rgb_camera_info_cb(self, cameraInfo):
    try:
      if self.rgb_intrinsics:
        return
      self.rgb_intrinsics = {"width":0, "height":0,"fx":0,"fy":0,"cx":0,"cy":0}
      self.rgb_intrinsics['width'] = cameraInfo.width
      self.rgb_intrinsics['height'] = cameraInfo.height
      self.rgb_intrinsics['fx'] = cameraInfo.K[0]
      self.rgb_intrinsics['fy'] = cameraInfo.K[4]
      self.rgb_intrinsics['cx'] = cameraInfo.K[2]
      self.rgb_intrinsics['cy'] = cameraInfo.K[5]
      print(self.rgb_intrinsics)
  
    except CvBridgeError as e:
        print(e)
        return



  def slam_cb(self,data):
    self.slam_flag = True
    
    self.drone_x = data.pose.pose.position.x
    self.drone_y = data.pose.pose.position.y

    orientation_q = data.pose.pose.orientation
    orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
    (self.roll, self.pitch, self.yaw) = euler_from_quaternion(orientation_list)


  def gs_yolo_cb(self,data):
    if not(self.depth_intrinsics and self.rgb_intrinsics):
      return
    print(len(data.results),self.depth_flag, self.slam_flag)
    if data.results and (self.depth_flag == True) and (self.slam_flag == True):

      for idx in range(len(data.results)):

        # Get center of box
        self.center_of_box = (data.results[idx].x_center,data.results[idx].y_center)
        x_min = data.results[idx].xmin
        x_max = data.results[idx].xmax
        y_min = data.results[idx].ymin
        y_max = data.results[idx].ymax
        x_depth_min, x_depth_max, y_depth_min, y_depth_max = self.reproject_rgb_to_depth(x_min, x_max, y_min, y_max)

        croped_image = self.cv_depth_image[y_depth_min:y_depth_max,x_depth_min:x_depth_max]
        print(np.min(self.cv_depth_image))
        print("maximum")
        print(np.max(self.cv_depth_image))
        
        result_img = cv2.cvtColor(self.cv_depth_image*500, cv2.COLOR_GRAY2BGR)
        result_img = cv2.rectangle(result_img, (x_min,y_min), (x_max, y_max),(0,0,0), 5)

        cv2.imshow("depth_image",result_img)
        cv2.waitKey(1)
        depth = np.median(croped_image)

        print("median : ", depth)

        self.depth_value_gaussian = depth

        object_x, object_y = self.get_object_pos(data.results[idx].x_center,data.results[idx].y_center,self.depth_value_gaussian)

        self.depth_info.x = object_x
        self.depth_info.y = object_y
        self.depth_info.label = data.results[idx].label
        self.depth_info_pub.publish(self.depth_info)
        
        
        print("====")
        print(self.center_of_box)
        




  def depth_cb(self,data):
    if not self.depth_intrinsics:
      return
    self.depth_flag = True
    self.cv_depth_image = self.bridge.imgmsg_to_cv2(data, data.encoding)

  def yolo_cb(self,data):
    if not(self.depth_intrinsics and self.rgb_intrinsics):
      return
    print(len(data.results),self.depth_flag, self.slam_flag)
    if data.results and (self.depth_flag == True) and (self.slam_flag == True):

      for idx in range(len(data.results)):

        # Get center of box
        self.center_of_box = (data.results[idx].x_center,data.results[idx].y_center)

        print("====")
        print(self.center_of_box)
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
        print("min:",depth_min)
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



  def get_object_pos(self,u,v,z):

    fx = self.rgb_intrinsics["fx"]
    fy = self.rgb_intrinsics["fy"]
    cx = self.rgb_intrinsics["cx"]
    cy = self.rgb_intrinsics["cy"]
   
    print("(u,v) : ", u,v)
    #Pc
    print('z : ', z)
    x,y,z = z*np.linalg.inv(np.matrix([[fx,0,cx],[0,fy,cy],[0,0,1]])) * np.matrix([[u],[v],[1]]) / 1000.0
    # temp_x = x
    # temp_y = y
    # temp_z = z
    # x = temp_z
    # y = -temp_y
    # z = temp_x
    try:
      # (trans, rot) = self.listener.lookupTransform('/camera_d435', 'world', rospy.Time(0))
      (trans, rot) = self.listener.lookupTransform('/map', '/camera_color_optical_frame', rospy.Time(0))
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
      print("except")
    # print('x y z : ', x,y,z)
    
    # get camera static tf and certify 
    # get transformation matrix Tmc 
    
    world_cam_T = tf.transformations.translation_matrix((trans[0],trans[1],trans[2]))
    world_cam_R = tf.transformations.quaternion_matrix((rot[0],rot[1],rot[2],rot[3]))
    # generate TF matrix in python(Tmc)
    Tmc = np.matmul(world_cam_T,world_cam_R)

    Pm = np.matmul(Tmc, np.array([[x.item()], [y.item()], [z.item()], [1]]))

    Pm_pub = PoseStamped()

    Pm_pub.header.stamp = rospy.Time.now()
    # Pm_pub.pose.position.x = Pm[0] / 1000.0
    # Pm_pub.pose.position.y = Pm[1] / 1000.0
    # Pm_pub.pose.position.z = Pm[2] / 1000.0
    Pm_pub.pose.position.x = Pm[0]
    Pm_pub.pose.position.y = Pm[1]
    Pm_pub.pose.position.z = Pm[2]
    
    self.draw_in_rviz(Pm_pub.pose.position.x,Pm_pub.pose.position.y, Pm_pub.pose.position.z)

    
    # self.object_estimator.publish(Pm)
    return Pm_pub.pose.position.x, Pm_pub.pose.position.y

  def draw_in_rviz(self,x,y,z):
    self.marker = Marker()

    self.marker.header.frame_id = "/camera_color_optical_frame"
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
    self.marker.color.r = np.random.randint(255)
    self.marker.color.g = np.random.randint(255)
    self.marker.color.b = np.random.randint(255)

    # self.marker.mesh_resource = "package://pr2_description/meshes/base_v0/base.dae"
    self.vis_pub.publish(self.marker)
    # rospy.loginfo("marker has been published")
  def draw_top_view(self, x, y,r,g,b):

    
    x += 250
    y += 250

    img = cv2.circle(self.img, (int(x),int(y)), 1 ,(r,g,b), -1)

    cv2.imshow('img',img)
    cv2.waitKey(1)

    
    
  def get_depth(self,x,y): # x y in rgb image
    x = int(x * width_factor)
    y = int(y * height_factor)
    return self.cv_depth_image[y][x]
  
  # def rgb_cb(self,data):
  #   try:
  #     #(480,640,3)

  #     self.cv_rgb_image = self.bridge.imgmsg_to_cv2(data,'bgr8')
  #     self.rgb_input = True
  #   except CvBridgeError as e:
  #     self.rgb_input = False
  #     rospy.logerr(e)

  def reproject_rgb_to_depth(self,x_min,x_max,y_min,y_max,margin=0):

    # regular coordinate system
    u_min = x_min - self.rgb_intrinsics["cx"]
    u_max = x_max - self.rgb_intrinsics["cx"]
    v_min = y_min - self.rgb_intrinsics["cy"]
    v_max = y_max - self.rgb_intrinsics["cy"]

    # transform rgb grid to regular depth grid
    u_depth_min = u_min*self.depth_intrinsics["fx"]/self.rgb_intrinsics["fx"] + margin
    u_depth_max = u_max*self.depth_intrinsics["fx"]/self.rgb_intrinsics["fx"] - margin
    v_depth_min = v_min*self.depth_intrinsics["fy"]/self.rgb_intrinsics["fy"] + margin
    v_depth_max = v_max*self.depth_intrinsics["fy"]/self.rgb_intrinsics["fy"] - margin
    
    # To make integer,To prevent exceeding image coordinate
    x_depth_min = max(int(u_depth_min + self.depth_intrinsics["cx"]),0)
    x_depth_max = min(self.depth_intrinsics["width"]-1, int(u_depth_max + self.depth_intrinsics["cx"]))
    y_depth_min = max(int(v_depth_min + self.depth_intrinsics["cy"]),0)
    y_depth_max = min(self.depth_intrinsics["height"]-1, int(v_depth_max + self.depth_intrinsics["cy"]))

    return x_depth_min, x_depth_max, y_depth_min, y_depth_max




def main(args):

  depth_camera_topic = "/camera/depth/image_rect_raw"
  depth_camera_info_topic = "/camera/depth/camera_info"
  yolo_output_topic = "/yolo_output"
  rgb_camera_info_topic = "/camera/color/camera_info"
  odometry_topic = "/odometry/filtered"

  print("depth_camera_topic = ", depth_camera_info_topic)

  rospy.init_node('depth_subscriber', anonymous=True)
  ic = image_converter(depth_camera_topic , depth_camera_info_topic, yolo_output_topic, rgb_camera_info_topic, odometry_topic)
 
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)