import numpy as np
import math


class Pure_pursuit():
    def __init__(self, look_a_head, velocity):
        self.ld = look_a_head
        self.vel = velocity

    def find_furthest_point(self, current, path):
        distances = np.sqrt(np.sum((np.array(path) - np.array(list(current))) ** 2, axis= 1))
        #print(distances)
        mask = distances <= self.ld
        possible_points = np.array(path)[mask]
        if len(possible_points) == 0:
            return None
        distances_within_ld = distances[distances <= self.ld]
        # Find the index of the furthest point within look ahead distance
        furthest_point_index = np.argmax(distances_within_ld)
        # Return the furthest point within look ahead distance
        return possible_points[furthest_point_index]
    
    def pure_pursuit_motion(self, current, current_heading, path):
        closest_point = self.find_furthest_point(current, path)
        #closest_point = path.pop()
        print('go to ', closest_point)
        if closest_point is not None:
            target_heading = math.atan2(- closest_point[0] + current[0], closest_point[1] - current[1])
            desired_steering_angle = target_heading - current_heading - math.pi /2
            print("turn", desired_steering_angle/math.pi * 180)
        else:
            target_heading = math.atan2(- path[-1][0] + current[0], path[-1][1] - current[1])
            desired_steering_angle = target_heading - current_heading - math.pi /2
        if desired_steering_angle > math.pi:
            desired_steering_angle -= 2 * math.pi
        elif desired_steering_angle < -math.pi:
            desired_steering_angle += 2 * math.pi
        if desired_steering_angle > math.pi/6 or desired_steering_angle < -math.pi/6:
            sign = 1 if desired_steering_angle > 0 else -1
            desired_steering_angle = sign * math.pi/4
            self.vel = 0.0
        return self.vel,desired_steering_angle