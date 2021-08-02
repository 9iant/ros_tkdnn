#!/usr/bin/env python
from __future__ import print_function

# general library 
import roslib
#roslib.load_manifest('depth_sub')
import sys
import rospy
import cv2
import std_msgs
import numpy as np
import pandas as pd

# geometry related message and module.
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from nav_msgs.msg import Odometry
import tf
from tf.transformations import *

# Camerea, image related message.
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

# For visualization
from visualization_msgs.msg import Marker, MarkerArray

# User defined message. #
from ros_tkdnn.msg import yolo_coordinateArray
from ros_tkdnn.msg import depth_info, depth_infoArray

'''
TODO: Potential warning due to hard coding:

gs_yolo_callback -> marker frame id
get_object_pos -> look up table

'''
# Topics and FRAME NAME 

CAMERA_OPTIC_FRAME_NAME = '/camera_depth_optical_frame'
GLOBAL_FRAME_NAME = 'global'
# SUBS
depth_camera_topic = "/camera/depth/image_rect_raw"
depth_camera_info_topic = "/camera/depth/camera_info"
yolo_output_topic = "/yolo_output"
rgb_camera_info_topic = "/camera/color/camera_info"
odometry_topic = "/odometry/filtered"
# PUBS


# main class.
class image_converter:

  def __init__(self):

    self.data = pd.DataFrame(columns=['label','x','y'])

    '''
    SUBSCRIBERS
    '''
    # depth image subscriber.
    self.bridge = CvBridge()
    self.depth_sub = rospy.Subscriber(depth_camera_topic, Image,self.depth_cb)
    self.depth_flag = False

    # slam odometry subscriber.
    self.slam_odom_sub = rospy.Subscriber(odometry_topic, Odometry, self.slam_cb)
    self.slam_flag = False

    # To handle geometric transformation.
    self.listener = tf.TransformListener()

    # yolo result subscriber (Main part).
    self.yolo_sub = rospy.Subscriber(yolo_output_topic, yolo_coordinateArray,self.gs_yolo_cb)

    # callback funtion.. but need to run once. 
    self.depth_camera_info_sub = rospy.Subscriber(depth_camera_info_topic, CameraInfo, self.depth_camera_info_cb)
    self.depth_intrinsics = None # if there is no camera_info topic, make exception
    self.rgb_camera_info_sub = rospy.Subscriber(rgb_camera_info_topic, CameraInfo, self.rgb_camera_info_cb)
    self.rgb_intrinsics = None # if there is no camera_info topic, make exception

    '''
    PUBLISHERS
    '''
    # publisher: one for visualization, one for depth result.
    # they are integrated in gs_yolo_cb function.
    self.depth_infoArray_pub = rospy.Publisher("/depth_estimator/depth_infoarray", depth_infoArray, queue_size = 1)
    self.vis_pub = rospy.Publisher("/depth_estimator/object_global_position", MarkerArray, queue_size=1)

  #------------------------------------------------------COORDINATE TRANSFORM------------------------------------------------------

  # run only once. save depth camera intrinsics.
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

  # run only once. save rgb camera intrinsics.
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

  # change coordinate system based on camera intrinsic params (rgb frame -> depth frame)
  def reproject_rgb_to_depth(self,x_min,x_max,y_min,y_max):

    # regular coordinate system
    u_min = x_min - self.rgb_intrinsics["cx"]
    u_max = x_max - self.rgb_intrinsics["cx"]
    v_min = y_min - self.rgb_intrinsics["cy"]
    v_max = y_max - self.rgb_intrinsics["cy"]

    # transform rgb grid to regular depth grid
    u_depth_min = u_min*self.depth_intrinsics["fx"]/self.rgb_intrinsics["fx"]
    u_depth_max = u_max*self.depth_intrinsics["fx"]/self.rgb_intrinsics["fx"]
    v_depth_min = v_min*self.depth_intrinsics["fy"]/self.rgb_intrinsics["fy"]
    v_depth_max = v_max*self.depth_intrinsics["fy"]/self.rgb_intrinsics["fy"] 
    
    # To make integer,To prevent exceeding image coordinate
    x_depth_min = max(int(u_depth_min + self.depth_intrinsics["cx"]),0)
    x_depth_max = min(self.depth_intrinsics["width"]-1, int(u_depth_max + self.depth_intrinsics["cx"]))
    y_depth_min = max(int(v_depth_min + self.depth_intrinsics["cy"]),0)
    y_depth_max = min(self.depth_intrinsics["height"]-1, int(v_depth_max + self.depth_intrinsics["cy"]))

    return x_depth_min, x_depth_max, y_depth_min, y_depth_max

  # using tf library, make global position.
  def get_object_pos(self,u,v,z):

    fx = self.depth_intrinsics["fx"]
    fy = self.depth_intrinsics["fy"]
    cx = self.depth_intrinsics["cx"]
    cy = self.depth_intrinsics["cy"]
    
    x = (u-cx)*z/fx/1000.0
    y = (v-cy)*z/fy/1000.0
    z = z/1000.0

    try:
      (trans, rot) = self.listener.lookupTransform(GLOBAL_FRAME_NAME, CAMERA_OPTIC_FRAME_NAME, rospy.Time(0))
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
      print("except")
    
    world_cam_T = tf.transformations.translation_matrix((trans[0],trans[1],trans[2]))
    world_cam_R = tf.transformations.quaternion_matrix((rot[0],rot[1],rot[2],rot[3]))
    # generate TF matrix in python(Tmc)
    Tmc = np.matmul(world_cam_T,world_cam_R)

    Pm = np.matmul(Tmc, np.array([[x.item()], [y.item()], [z.item()], [1]]))

    Pm_pub = PoseStamped()

    Pm_pub.header.stamp = rospy.Time.now()
    Pm_pub.pose.position.x = Pm[0]
    Pm_pub.pose.position.y = Pm[1]
    Pm_pub.pose.position.z = Pm[2]
    
    return Pm_pub.pose.position.x, Pm_pub.pose.position.y, Pm_pub.pose.position.z

  #---------------------------------------------------------SAVE DATA------------------------------------------------------

  # save depth image when it's published.
  def depth_cb(self,depth_image_topic):
    if not self.depth_intrinsics:
      return
    self.depth_flag = True
    self.cv_depth_image = self.bridge.imgmsg_to_cv2(depth_image_topic, depth_image_topic.encoding)


  # save slam odometry when it's publisehd.
  def slam_cb(self,odometry_topic):
    self.slam_flag = True
    
    self.drone_x = odometry_topic.pose.pose.position.x
    self.drone_y = odometry_topic.pose.pose.position.y

    orientation_q = odometry_topic.pose.pose.orientation
    orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
    (self.roll, self.pitch, self.yaw) = euler_from_quaternion(orientation_list)

  #----------------------------------------------------------MAIN CALLBACK-------------------------------------------------

  # main part of this node. if there is yolo output, generate depth estimation results.
  def gs_yolo_cb(self, yolo_array):
    if not(self.depth_intrinsics and self.rgb_intrinsics):
      return
    
    # Command window display. 
    obj_list = [a_result.label for a_result in yolo_array.results]
    print("Number of object : {0} // {1}".format(len(yolo_array.results), obj_list))
    print("Flags : Depth {0}, Odom {1}".format(self.depth_flag, self.slam_flag))
    print("-"*20)

    redcolor = np.random.rand() # for visualization part

    if yolo_array.results and self.depth_flag and self.slam_flag:

      depth_data = depth_infoArray()
      globalposition_data = MarkerArray()

      for idx in range(len(yolo_array.results)):

        # Get center of box
        x_min = yolo_array.results[idx].xmin
        x_max = yolo_array.results[idx].xmax
        y_min = yolo_array.results[idx].ymin
        y_max = yolo_array.results[idx].ymax
        x_depth_min, x_depth_max, y_depth_min, y_depth_max = self.reproject_rgb_to_depth(x_min, x_max, y_min, y_max)
        x_depth_center = (x_depth_max + x_depth_min)/2
        y_depth_center = (y_depth_max + y_depth_min)/2
        croped_image = self.cv_depth_image[y_depth_min:y_depth_max,x_depth_min:x_depth_max]
        
        # For debugging
        # result_img = cv2.cvtColor(self.cv_depth_image*100, cv2.COLOR_GRAY2BGR)
        # result_img = cv2.rectangle(result_img, (x_depth_min,y_depth_min), (x_depth_max, y_depth_max),(0,0,0), 5)

        # cv2.imshow("depth_image",result_img)
        # cv2.waitKey(1)
        depth = np.median(croped_image)

        object_x, object_y, object_z = self.get_object_pos(x_depth_center, y_depth_center , depth) # --> draw rviz pub 

        # depth estimator result
        depth_info_data = depth_info()
        depth_info_data.x = object_x
        depth_info_data.y = object_y
        depth_info_data.label = yolo_array.results[idx].label
        depth_info_data.confidence = yolo_array.results[idx].confidence
        # depth result array
        depth_data.results.append(depth_info_data)

        # visualization part
        marker = Marker()
        marker.header.frame_id = GLOBAL_FRAME_NAME
        marker.header.stamp = rospy.Time.now()
        marker.ns = "my_namespace"
        marker.id = idx
        marker.type = 2
        marker.action = Marker.ADD
        marker.pose.position.x = object_x
        marker.pose.position.y = object_y
        marker.pose.position.z = object_z
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.color.a = 0.5 # Don't forget to set the alpha!
        marker.color.r = redcolor
        marker.color.g = 0
        marker.color.b = 0
        # marker array
        globalposition_data.markers.append(marker) 

      self.depth_infoArray_pub.publish(depth_data)
      self.vis_pub.publish(globalposition_data) 
  
  
  #-----------------------------------------------------------------------------------------------------------------------


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
