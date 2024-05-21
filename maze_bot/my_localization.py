'''
Define the top-view picture into grid lines
Using Canny edge detection, fill in countours
Convert white pixels into black and reverse
Which grid cells have a black pixels will be occupied
Robot is attached which an Aruco marker for localizing its position

Map is represented in a 2D numpy array:
    0: unoccupied
    1: occupied
    2: robot's position
'''


import cv2
import numpy as np
import math
import cv2.aruco as aruco
from scipy.ndimage import binary_dilation

class Map:
    def __init__(self, original_image):
        self.image = original_image.copy()
        self.grid_data = None
        self.robot_pos = None
        self.y_size = 0
        self.x_size = 0
        self.detected = False
        self.robot_radius = 4
        self.safety_radius = 2


    #return top left and bottom right of the rectangle boundary of the robot
    def robot_localization(self):
        status = None
        top_left = None
        bottom_right = None
        temp_img = self.image.copy()
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        parameters = cv2.aruco.DetectorParameters()
        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(temp_img, dictionary, parameters=parameters)

        if  not markerCorners:
            status = False
        # Draw rectangles around the detected markers
        for corners in markerCorners:
            status = True
            # The corners are in the order top-left, top-right, bottom-right, bottom-left
            top_left, top_right, bottom_right, bottom_left = corners[0]
            # Convert the corners to integers
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))
        return status, top_left, bottom_right
    
    def pixel_to_grid(self, pixel):
        print('goal',pixel[0]//10, pixel[1]//10)
        return (pixel[0]//10, pixel[1]//10)

        
    def mapping_update(self, temp_goal):
        #get robot boundary
        status, top_left, bottom_right = self.robot_localization() 
        if status == True:
            self.detected = True
            # Convert the image to grayscale
            gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)

            # Apply Canny edge detection
            edges = cv2.Canny(gray,150,200)        

            # Define the kernel for dilation
            kernel = np.ones((5,5), np.uint8)

            # Apply dilation
            dilated = cv2.dilate(edges, kernel, iterations=1)

            # Find contours
            contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Create a copy of the image to draw filled contours on
            filled_image = gray.copy()

            for contour in contours:
                # Fill the contour on the image
                cv2.drawContours(filled_image, [contour], 0, (255, 255, 255), -1)

            # Invert the colors
            inverted_image = cv2.bitwise_not(filled_image)

            # Define the size of the grid
            grid_size = 10  

            # Initialize counters for the grid coordinates
            grid_x = 0
            grid_y = 0

            center_x = (top_left[0] + bottom_right[0]) // 2
            center_y = (top_left[1] + bottom_right[1]) // 2


            #grid map 0------------------->x
            #         ' 
            #         ' 
            #         ' 
            #         ' 
            #         ' 
            #         ' 
            #         v y

            # Initialize an empty numpy array for the grid
            grid_data = np.zeros((math.ceil(inverted_image.shape[0]/grid_size), math.ceil(inverted_image.shape[1]/grid_size)), dtype=np.uint8)


            for i in range(0, inverted_image.shape[0], grid_size):          #y
                grid_x = 0
                for j in range(0, inverted_image.shape[1], grid_size):      #x
                    # Get the current grid
                    grid = inverted_image[i:i+grid_size, j:j+grid_size]

                    if grid_data[grid_y, grid_x] != 2:
                        # Check if there are any black pixels in the grid
                        if np.any(grid == 0):
                            grid_data[i//grid_size, j//grid_size] = 1
                        else:
                            pass           
                    # Increment the grid x-coordinate
                    grid_x += 1
                # Increment the grid y-coordinate
                grid_y += 1
            

            # Iterate over the cells within the radius
            for i in range(-self.robot_radius , self.robot_radius  + 1):
                for j in range(-self.robot_radius , self.robot_radius  + 1):
                    # Check if the cell is within the grid
                    if (0 <= center_y // grid_size + i  < grid_data.shape[0]) and (0 <= center_x // grid_size + j < grid_data.shape[1]):
                        # Set the cell value to 0
                        grid_data[center_y // grid_size + i, center_x // grid_size + j] = 0
            grid_data[center_y // grid_size, center_x // grid_size] = 2
            self.robot_pos = (center_y // grid_size, center_x // grid_size)

            #update safety radius
            structuring_element = np.ones((2*self.safety_radius+1, 2*self.safety_radius+1))
            temp_grid = np.copy(grid_data)
            grid_data[grid_data == 2] = 0
                # Use binary dilation to find the neighbors
            dilated = binary_dilation(grid_data, structure=structuring_element)
            grid_data = np.maximum(temp_grid, dilated)

            dimensions = grid_data.shape
            self.y_size = dimensions[0]
            self.x_size = dimensions[1]

            
            #draw map
            grid_x = 0
            grid_y = 0
            for i in range(0, inverted_image.shape[0], grid_size):          #y
                grid_x = 0
                for j in range(0, inverted_image.shape[1], grid_size):      #x
                    if grid_data[grid_y, grid_x] == 1:
                        # Check if there are any black pixels in the grid
                            # Draw X
                        cv2.line(inverted_image, (j+grid_size//4, i+grid_size//4), (j+3*grid_size//4, i+3*grid_size//4), (255, 255, 255), 1)
                        cv2.line(inverted_image, (j+3*grid_size//4, i+grid_size//4), (j+grid_size//4, i+3*grid_size//4), (255, 255, 255), 1)
                    elif grid_data[grid_y, grid_x] == 2:
                        #pass
                        # Draw O
                        cv2.circle(inverted_image, (j+grid_size//2, i+grid_size//2), grid_size//3, (255, 255, 255), 1)
                    else:
                        pass

                    grid_x += 1

                # Increment the grid y-coordinate
                grid_y += 1
        
            self.grid_data = grid_data
            return inverted_image
            