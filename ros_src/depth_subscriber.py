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

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from ros_tkdnn.msg import yolo_coordinateArray

import numpy as np

class image_converter:

  def __init__(self):
  

    
    self.bridge = CvBridge()
    self.depth_sub = rospy.Subscriber("/camera/depth/image_rect_raw",Image,self.depth_cb)
    self.rgb_sub = rospy.Subscriber("/camera/color/image_raw",Image,self.rgb_cb)
    # subscribe yolo output
    self.yolo_sub = rospy.Subscriber("/yolo_output",yolo_coordinateArray,self.yolo_cb)

    self.depth_gaussian_pub = rospy.Publisher("/depth_gaussian", UInt16, queue_size = 10)
    self.depth_pub = rospy.Publisher("/depth_center", UInt16, queue_size = 10)
    self.depth_image_pub = rospy.Publisher("/depth_image_pub", Image, queue_size = 10)
   
  def depth_cb(self,data):
    try:
      #(480,640,3)
      self.cv_depth_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
      
    except CvBridgeError as e:
      rospy.logerr(e)
    

  def yolo_cb(self,data):
    
  
    
    if len(data.results) > 0:
      #print('hi')


      # make sure the number of results
      self.yolo_output_list = []
      for idx in range(len(data.results)):
      
        self.yolo_output_list.append(
          np.array([
        data.results[idx].x_center, #0
        data.results[idx].y_center, #1
        data.results[idx].w, # 2
        data.results[idx].h, #3
        data.results[idx].xmin, #4
        data.results[idx].xmax, #5
        data.results[idx].ymin, #6
        data.results[idx].ymax, #7
        ] ,
        dtype=np.int32

        ))

        
        
        


    else:

      self.yolo_output_list = []
    #  rospy.logwarn("No results")
      

    self.draw()
    self.depth_gaussian_pub.publish(self.depth_value_gaussian)
    self.depth_pub.publish(self.depth_value)
    self.depth_image_pub.publish(CvBridge().cv2_to_imgmsg(self.cv_rgb_image, encoding="bgr8"))
          
        

    
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



    

  def draw(self):
    
    
    self.color = (255,255,255)

    for idx in range(len(self.yolo_output_list)):
      
      # Draw Bounding box
      self.cv_rgb_image = cv2.rectangle(
        self.cv_rgb_image, 
        (self.yolo_output_list[idx][4],self.yolo_output_list[idx][6]),
        (self.yolo_output_list[idx][5],self.yolo_output_list[idx][7]),

        self.color,
        3)

      # Get center of box
      self.cecnter_of_box = (self.yolo_output_list[idx][0],self.yolo_output_list[idx][1])

      # Draw center of box 
      self.cv_rgb_image = cv2.circle(
        self.cv_rgb_image, self.cecnter_of_box , 3, self.color, -1)

      # Gaussian Sampling for robust depth estimation
      num_samples = 1000
      
      self.gaussian_sampling = np.random.multivariate_normal(self.cecnter_of_box,
        [[self.yolo_output_list[idx][2],0],[0,self.yolo_output_list[idx][3]]], num_samples)

      # Get depth each sampling points

      self.depth_value_list = []

      point_x = []
      point_y = []

      for point in self.gaussian_sampling:

      
        self.depth_value_list.append(self.get_depth( int(point[0]), int(point[1])))  
        self.cv_rgb_image = cv2.circle(self.cv_rgb_image, (int(point[0]), int(point[1])), 3, self.color, -1)

        point_x.append(int(point[0]))
        point_y.append(int(point[1]))

      # Choose representative point 
      self.depth_value = self.cv_depth_image[self.cecnter_of_box[0]][self.cecnter_of_box[1]]#[self.yolo_output_list[idx][1]][self.yolo_output_list[idx][0]]
      
      depth_min = np.max(self.depth_value_list)
      for depth in self.depth_value_list:
        
        if depth > 0 and depth < depth_min:
          depth_min = depth 

        if depth_min == 0:
          rospy.roswarn("minimum depth is 0, ERROR")
        

      self.depth_value_gaussian = depth_min
      self.point_x = int(np.mean(point_x))
      self.point_y = int(np.mean(point_y))
      print(depth_min)
      self.cv_rgb_image = cv2.circle(
        self.cv_rgb_image, (self.point_x, self.point_y), 7, (0,255,0), -1)


      cv2.putText(self.cv_rgb_image, str(self.depth_value_gaussian), (self.yolo_output_list[idx][4],self.yolo_output_list[idx][6]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, self.color, 2, cv2.LINE_AA)
      #cv2.putText(self.cv_rgb_image, str(self.depth_value), (self.yolo_output_list[idx][4],self.yolo_output_list[idx][6]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, self.color, 2, cv2.LINE_AA)

   
    # cv2.imshow("RGB window", self.cv_rgb_image)
    # cv2.waitKey(3)



def main(args):
  ic = image_converter()
  rospy.init_node('depth_subscriber', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
