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

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from ros_tkdnn.msg import yolo_coordinateArray
from ros_tkdnn.msg import depth_info
from tf.transformations import *
import numpy as np

class image_converter:

  def __init__(self):
    
   # self.cv_depth_image = None

    self.bridge = CvBridge()
    self.depth_sub = rospy.Subscriber("/camera/depth/image_rect_raw",Image,self.depth_cb)
    self.yolo_sub = rospy.Subscriber("/yolo_output",yolo_coordinateArray,self.yolo_cb)
    self.depth_gaussian_pub = rospy.Publisher("/depth_gaussian", depth_info, queue_size = 1)
    self.depth_info = depth_info()

    self.slam_odom = rospy.Subscriber("/ov_msckf/poseimu", PoseWithCovarianceStamped, self.slam_cb)
   # self.rgb_sub = rospy.Subscriber("/camera/color/image_raw",Image,self.rgb_cb)
    # subscribe yolo output
    
   # self.depth_pub = rospy.Publisher("/depth_center", UInt16, queue_size = 1)
   # self.depth_image_pub = rospy.Publisher("/depth_image_pub", Image, queue_size = 1)

    self.depth_flag = False

  def slam_cb(self,data):
    
    self.drone_x = data.pose.pose.position.x
    self.drone_y = data.pose.pose.position.y

    orientation_q = data.pose.pose.orientation
    orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
    (self.roll, self.pitch, self.yaw) = euler_from_quaternion(orientation_list)



    

  def depth_cb(self,data):
    
    self.depth_flag = True
    self.cv_depth_image = self.bridge.imgmsg_to_cv2(data, data.encoding)

  def yolo_cb(self,data):

    if data.results and (self.depth_flag == True):

      for idx in range(len(data.results)):

        # Get center of box
        self.cecnter_of_box = (data.results[idx].x_center,data.results[idx].y_center)

        # Gaussian sampling 
        num_samples = 1000
        
        self.gaussian_sampling = np.random.multivariate_normal(self.cecnter_of_box,
          [[data.results[idx].w,0],[0,data.results[idx].h]], num_samples)

        # Get depth each point

        self.depth_value_list = []

        point_x = []
        point_y = []

        for point in self.gaussian_sampling:

          self.depth_value_list.append(self.get_depth( int(point[0]), int(point[1])))  
    
          point_x.append(int(point[0]))
          point_y.append(int(point[1]))

        # Choose representative point (minimum depth)
      
        depth_min = np.max(self.depth_value_list)
        for depth in self.depth_value_list:
          
          if depth > 0 and depth < depth_min:
            depth_min = depth 

          if depth_min == 0:
            rospy.roswarn("minimum depth is 0, ERROR")
          

        self.depth_value_gaussian = depth_min
        # print("RUN")
        # Get Obstacle Position based on SLAM odom

        obstacle_x, obstacle_y = self.get_obstacle_pos(self.depth_value_gaussian, self.drone_x, self.drone_y, self.roll , self.pitch , self.yaw)



        # self.depth_info.depth = self.depth_value_gaussian
        # self.depth_info.label = data.results[idx].label
        # self.depth_info.x_center = data.results[idx].x_center
        # self.depth_info.y_center = data.results[idx].y_center

       # rospy.loginfo(self.depth_info)
      #   self.depth_gaussian_pub.publish(self.depth_info) 


      #   self.point_x = int(np.mean(point_x))
      #   self.point_y = int(np.mean(point_y))
        
      #   self.cv_rgb_image = cv2.circle(
      #     self.cv_rgb_image, (self.point_x, self.point_y), 7, (0,255,0), -1)

      # self.depth_image_pub.publish(CvBridge().cv2_to_imgmsg(self.cv_rgb_image, encoding="bgr8"))

    else:
      pass
      #rospy.logwarn("no detection")
            
  def get_obstacle_pos (self,depth_gaussian,drone_x,drone_y,roll,pitch,yaw):
    
   
    depth_z = depth_gaussian * np.cos(pitch)

    obstacle_x = drone_x + depth_z * np.cos(yaw)
    obstacle_y = drone_y + depth_z * np.sin(yaw)

    return obstacle_x, obstacle_y
    
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
