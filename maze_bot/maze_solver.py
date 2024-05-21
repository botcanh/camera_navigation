import rclpy
import time
from rclpy.node import Node
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2
from math import pi,cos,sin
from math import pow , atan2,sqrt , degrees,asin, radians

from .my_path_planning import Dijkstra
from .my_localization import Map

from nav_msgs.msg import Odometry
from std_msgs.msg import String

import numpy as np
from numpy import interp
from .bot_pure_pursuit import Pure_pursuit


class maze_solver(Node):

    def __init__(self):
        
        super().__init__("maze_solving_node")
        
        self.velocity_publisher = self.create_publisher(Twist,'/cmd_vel',10)
        self.videofeed_subscriber = self.create_subscription(Image,'/upper_camera/image_raw',self.get_video_feed_cb,10)
        self.videofeed_subscriber_kitchen = self.create_subscription(Image,'/upper_camera_kitchen/image_raw',self.get_video_feed_cb_kitchen,10)
        self.videofeed_subscriber_bedroom = self.create_subscription(Image,'/upper_camera_bedroom/image_raw',self.get_video_feed_cb_bedroom,10)
        
        # Visualizing what the robot sees by subscribing to bot_camera/Image_raw
        self.timer = self.create_timer(0.2, self.maze_solving)
        self.bridge = CvBridge()
        self.vel_msg = Twist()
        self.image_count = 1

        
        # Creating objects for each stage of the robot navigation
        self.bot_localizer_mapper = None
        self.bot_localizer_mapper_kitchen = None
        self.bot_localizer_mapper_bedroom = None
        self.bot_pathplanner = None
        self.bot_motion_control = None

        # Subscrbing to receive the robot pose in simulation
        #self.pose_subscriber = self.create_subscription(Odometry,'/odom',self.bot_motionplanner.get_pose,10)
        self.vel_subscriber = self.create_subscription(Odometry,'/odom',self.get_pose,10)
        self.bot_speed = 0
        self.bot_turning = 0
        self.yaw = 0
        self.desired_yaw = 0
        self.left = False
        self.right = False
        self.temp_goal = (0,0)

        self.sat_view = None
        self.place = None
        self.goal_place = None
        self.goal = (0,0)

               
    def get_video_feed_cb(self,data):
        frame = self.bridge.imgmsg_to_cv2(data,'bgr8')
        self.sat_view = frame

    def get_video_feed_cb_kitchen(self,data):
        frame = self.bridge.imgmsg_to_cv2(data,'bgr8')
        self.sat_view_kitchen = frame
    
    def get_video_feed_cb_bedroom(self,data):
        frame = self.bridge.imgmsg_to_cv2(data,'bgr8')
        self.sat_view_bedroom = frame

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
  
    def get_pose(self,data):

        # We get the bot_turn_angle in simulation Using same method as Gotogoal.py
        quaternions = data.pose.pose.orientation
        (roll,pitch,yaw)=self.euler_from_quaternion(quaternions.x, quaternions.y, quaternions.z, quaternions.w)
        yaw_deg = degrees(yaw)
        self.yaw = yaw_deg

        # [Maintaining the Consistency in Angle Range]
        #if (yaw_deg>0):
        #    self.yaw = yaw_deg
        #else:
            # -160 + 360 = 200, -180 + 360 = 180 . -90 + 360 = 270
        #    self.yaw = yaw_deg + 360
        
        #              Bot Rotation 
        #      (OLD)        =>      (NEW) 
        #   [-180,180]             [0,360]


    def get_bot_speed(self,data):
        self.bot_speed = (data.twist.twist.linear.x)
        if self.bot_speed<0.1:
            self.bot_speed = 0.05

        self.bot_turning = data.twist.twist.angular.z

    def motion_stop(self, bot_mapper):
        pics = bot_mapper.mapping_update(self.temp_goal) 
        cv2.imshow("map", pics)
        self.vel_msg.linear.x = 0.0
        self.vel_msg.linear.y = 0.0
        self.vel_msg.linear.z = 0.0

        self.vel_msg.angular.x = 0.0
        self.vel_msg.angular.y = 0.0
        self.vel_msg.angular.z = 0.0
        self.velocity_publisher.publish(self.vel_msg)
              
    def click_event(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.goal = (y,x)
            self.goal_place = 'living_room'
        print('clicked living room', self.goal)

    def click_event_bedroom(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.goal = (y,x)
            self.goal_place = 'bedroom'
        print('clicked bedroom', self.goal)

    def click_event_kitchen(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.goal = (y,x)
            self.goal_place = 'kitchen'
        print('clicked kitchen', self.goal)

    def update_desired_yaw(self, current_pos, desired_pos):
        if (desired_pos[0] - current_pos[0] == -1) and (desired_pos[1] - current_pos[1] == 0):
            self.desired_yaw = 0
        elif (desired_pos[0] - current_pos[0] == 1) and (desired_pos[1] - current_pos[1] == 0):
            self.desired_yaw = 180
        elif (desired_pos[0] - current_pos[0] == 0) and (desired_pos[1] - current_pos[1] == 1):
            self.desired_yaw = 270
        elif (desired_pos[0] - current_pos[0] == 0) and (desired_pos[1] - current_pos[1] == -1):
            self.desired_yaw = 90
        elif (desired_pos[0] - current_pos[0] == -1) and (desired_pos[1] - current_pos[1] == 1):
            self.desired_yaw = 315
        elif (desired_pos[0] - current_pos[0] == -1) and (desired_pos[1] - current_pos[1] == -1):
            self.desired_yaw = 45
        elif (desired_pos[0] - current_pos[0] == 1) and (desired_pos[1] - current_pos[1] == -1):
            self.desired_yaw = 135
        elif (desired_pos[0] - current_pos[0] == 1) and (desired_pos[1] - current_pos[1] == 1):
            self.desired_yaw = 225

    def maze_solving(self):
        #cv2.imshow('raw', self.bot_view_raw)
        #cv2.imshow('depth_image',self.bot_view_depth)

        frame = self.sat_view
        frame_kitchen = self.sat_view_kitchen
        frame_bedroom = self.sat_view_bedroom
        cv2.imshow("original", self.sat_view)
        cv2.imshow("original_bedroom", self.sat_view_bedroom)
        cv2.imshow("original_kitchen", self.sat_view_kitchen)

        self.bot_localizer_mapper = Map(frame)
        self.bot_localizer_mapper_kitchen = Map(frame_kitchen)
        self.bot_localizer_mapper_bedroom = Map(frame_bedroom)

        map = self.bot_localizer_mapper.mapping_update(self.temp_goal)
        map_kitchen = self.bot_localizer_mapper_kitchen.mapping_update(self.temp_goal)
        map_bedroom = self.bot_localizer_mapper_bedroom.mapping_update(self.temp_goal)

        #cv2.imshow("map", map)
        #cv2.imshow("map kitchen", map_kitchen)
        #cv2.imshow("map bedroom", map_bedroom)
        
        cv2.setMouseCallback('original', self.click_event)
        cv2.setMouseCallback('original_bedroom', self.click_event_bedroom)
        cv2.setMouseCallback('original_kitchen', self.click_event_kitchen)

        if self.bot_localizer_mapper.detected == True:
            self.place = 'living_room'
            cv2.imshow("map", map)
        if self.bot_localizer_mapper_kitchen.detected == True:
            self.place = 'kitchen'
            cv2.imshow("map kitchen", map_kitchen)
        if self.bot_localizer_mapper_bedroom.detected == True:
            self.place = 'bedroom' 
            cv2.imshow("map bedroom", map_bedroom)

        
        if self.goal == (0,0):
            print("waiting for goal")
        else:
            print(self.place)
            if self.place == 'living_room':
                #print(self.bot_localizer_mapper.robot_pos) #(36, 63)(y,x)
                #print(self.bot_localizer_mapper.x_size, self.bot_localizer_mapper.y_size)
                temp = self.goal
                if self.goal_place == 'kitchen':
                    self.goal = (9,816)
                elif self.goal_place == 'bedroom':
                    self.goal = (569, 639)
                self.bot_pathplanner = Dijkstra(self.bot_localizer_mapper.grid_data, self.bot_localizer_mapper.x_size, self.bot_localizer_mapper.y_size)
                self.bot_motion_control = Pure_pursuit(3, 0.3)
                path = self.bot_pathplanner.perform(self.bot_localizer_mapper.pixel_to_grid(self.goal)) #input as (y,x)
                #print(path)
                if path is not None:
                    vel, steering = self.bot_motion_control.pure_pursuit_motion(self.bot_localizer_mapper.robot_pos, radians(self.yaw), path)
                    self.vel_msg.linear.x = vel
                    self.vel_msg.linear.y = 0.0
                    self.vel_msg.linear.z = 0.0

                    self.vel_msg.angular.x = 0.0
                    self.vel_msg.angular.y = 0.0
                    self.vel_msg.angular.z = steering
                    self.velocity_publisher.publish(self.vel_msg)
                if self.bot_pathplanner.reached == True:
                    if temp != self.goal:
                        self.vel_msg.linear.x = 0.3
                        self.vel_msg.linear.y = 0.0
                        self.vel_msg.linear.z = 0.0

                        self.vel_msg.angular.x = 0.0
                        self.vel_msg.angular.y = 0.0
                        self.vel_msg.angular.z = 0.0
                        self.velocity_publisher.publish(self.vel_msg)
                    else:
                        print("REACHED")
                        self.motion_stop(self.bot_localizer_mapper)
                        self.destroy_node()
                self.goal = temp
            elif self.place == 'bedroom':
                #print(self.bot_localizer_mapper_bedroom.robot_pos) #(36, 63)(y,x)
                #print(self.bot_localizer_mapper.x_size, self.bot_localizer_mapper.y_size)
                temp = self.goal
                if self.goal_place != 'bedroom':
                    self.goal = (4, 717)
                self.bot_pathplanner = Dijkstra(self.bot_localizer_mapper_bedroom.grid_data, self.bot_localizer_mapper_bedroom.x_size, self.bot_localizer_mapper_bedroom.y_size)
                self.bot_motion_control = Pure_pursuit(3, 0.3)
                path = self.bot_pathplanner.perform(self.bot_localizer_mapper_bedroom.pixel_to_grid(self.goal)) #input as (y,x)
                #print(path)
                if path is not None:
                    vel, steering = self.bot_motion_control.pure_pursuit_motion(self.bot_localizer_mapper_bedroom.robot_pos, radians(self.yaw), path)
                    self.vel_msg.linear.x = vel
                    self.vel_msg.linear.y = 0.0
                    self.vel_msg.linear.z = 0.0

                    self.vel_msg.angular.x = 0.0
                    self.vel_msg.angular.y = 0.0
                    self.vel_msg.angular.z = steering
                    self.velocity_publisher.publish(self.vel_msg)
                if self.bot_pathplanner.reached == True:
                    if temp != self.goal:
                        self.vel_msg.linear.x = 0.3
                        self.vel_msg.linear.y = 0.0
                        self.vel_msg.linear.z = 0.0

                        self.vel_msg.angular.x = 0.0
                        self.vel_msg.angular.y = 0.0
                        self.vel_msg.angular.z = 0.0
                        self.velocity_publisher.publish(self.vel_msg)
                    else:
                        print("REACHED")
                        self.motion_stop(self.bot_localizer_mapper)
                        self.destroy_node()
                self.goal = temp
            elif self.place == 'kitchen':
                temp = self.goal
                if self.goal_place != 'kitchen':
                    self.goal = (718, 719)
                #print(self.bot_localizer_mapper.x_size, self.bot_localizer_mapper.y_size)
                self.bot_pathplanner = Dijkstra(self.bot_localizer_mapper_kitchen.grid_data, self.bot_localizer_mapper_kitchen.x_size, self.bot_localizer_mapper_kitchen.y_size)
                self.bot_motion_control = Pure_pursuit(3, 0.3)
                path = self.bot_pathplanner.perform(self.bot_localizer_mapper_kitchen.pixel_to_grid(self.goal)) #input as (y,x)
                #print(path)
                if path is not None:
                    vel, steering = self.bot_motion_control.pure_pursuit_motion(self.bot_localizer_mapper_kitchen.robot_pos, radians(self.yaw), path)
                    self.vel_msg.linear.x = vel
                    self.vel_msg.linear.y = 0.0
                    self.vel_msg.linear.z = 0.0

                    self.vel_msg.angular.x = 0.0
                    self.vel_msg.angular.y = 0.0
                    self.vel_msg.angular.z = steering
                    self.velocity_publisher.publish(self.vel_msg)
                if self.bot_pathplanner.reached == True:
                    if temp != self.goal:
                        self.vel_msg.linear.x = 0.3
                        self.vel_msg.linear.y = 0.0
                        self.vel_msg.linear.z = 0.0

                        self.vel_msg.angular.x = 0.0
                        self.vel_msg.angular.y = 0.0
                        self.vel_msg.angular.z = 0.0
                        self.velocity_publisher.publish(self.vel_msg)
                    else:
                        print("REACHED")
                        self.motion_stop(self.bot_localizer_mapper)
                        self.destroy_node()
                self.goal = temp
        cv2.waitKey(1) 

def main(args =None):
    rclpy.init()
    node_obj =maze_solver()
    rclpy.spin(node_obj)
    rclpy.shutdown()


if __name__ == '__main__':
    main()