import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2

from nav_msgs.msg import Odometry
from std_msgs.msg import String

import numpy as np
from numpy import interp
from math import pi,cos,sin
from math import pow , atan2,sqrt , degrees,asin, radians


class robot_vision(Node):

    def __init__(self):
        
        super().__init__("maze_solving_node")
        
        # Visualizing what the robot sees by subscribing to bot_camera/Image_raw
        self.bot_subscriber = self.create_subscription(Image,'/depth_camera/depth/image_raw',self.process_data_depth,10)
        self.bot_view_subcriber = self.create_subscription(Image,'/depth_camera/image_raw',self.process_data_raw,10)
        self.timer = self.create_timer(0.2, self.robot_vision)
        self.bridge = CvBridge()
        self.image_count = 1

        self.sat_view = None
        self.place = None
        self.goal_place = None
        self.goal = (0,0)


    @staticmethod
    def euler_from_quaternion(x, y, z, w):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = atan2(t3, t4)

        return roll_x, pitch_y, yaw_z 
    
    def depthToCV8UC1(self, float_img):
        # Process images
        mono8_img = np.zeros_like(float_img, dtype=np.uint8)
        cv2.convertScaleAbs(float_img, mono8_img, alpha=10, beta=0.0)
        return mono8_img
    
    def process_data_depth(self, data):
        float_view = self.bridge.imgmsg_to_cv2(data, '8UC1')
        self.bot_view_depth = self.depthToCV8UC1(float_view)
        # Apply thresholding to create a binary image
        ret, thresh = cv2.threshold(self.bot_view_depth, 30, 50, 0)

        # Find contours in the binary image
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the original image and print depth data at the center of each contour
        for cnt in contours:
            # Calculate the moments to find the centroid
            M = cv2.moments(cnt)
            if M["m00"] != 0: # avoid division by zero
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0

            # Draw the contour and center of the shape on the image
            cv2.drawContours(self.bot_view_depth, [cnt], -1, (0, 255, 0), 2)
            cv2.circle(self.bot_view_depth, (cX, cY), 7, (255, 255, 255), -1)

            # Print the depth data at the center of the contour
            depth = float_view[cY, cX]
            cv2.putText(self.bot_view_depth, f"{depth}m", (cX + 5, cY + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow('depth_image', self.bot_view_depth)


    def process_data_raw(self, data):
      self.bot_view_raw = self.bridge.imgmsg_to_cv2(data, 'bgr8')
      cv2.imshow('raw', self.bot_view_raw)


    def robot_vision(self):
        cv2.waitKey(1) 

def main(args =None):
    rclpy.init()
    node_obj =robot_vision()
    rclpy.spin(node_obj)
    rclpy.shutdown()


if __name__ == '__main__':
    main()