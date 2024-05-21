import numpy as np
import math
import heapq

#pos : (y,x)

class Dijkstra:
    def __init__(self, grid_data, x_size, y_size):
        self.data = np.copy(grid_data)
        self.x_size = x_size
        self.y_size = y_size
        self.x_pos = 0
        self.y_pos = 0
        self.reached = False

    def get_init_pose(self):
        for i in range(0, self.y_size):
            for j in range(0, self.x_size):
                if self.data[i][j] == 2:
                    self.y_pos = i
                    self.x_pos = j
                    break

    def perform(self, goal):
        self.get_init_pose()
        start = (self.y_pos, self.x_pos)
        #print("start!")
        #print(self.x_size, self.y_size)
        neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
        close_set = set()
        came_from = {}
        gscore = {start:0}
        oheap = [] #openset
        heapq.heappush(oheap, (gscore[start], start)) # (set, (gscore, index))
        #print(start[1], start[0])

        while oheap:
            current = heapq.heappop(oheap)[1] #get the position
            #print(current[1], current[0])
            #trace path
            if current == goal:
                data = []
                while current in came_from:
                    data.append(current)
                    current = came_from[current]
                #data = data + [start]
                #data = data[::-1]
                if len(data) == 1:
                    self.reached = True # threshold
                return data

            close_set.add(current)
            for i, j in neighbors:
                neighbor = current[0] + j, current[1] + i
                if i + j != 1 and i + j != -1:
                    tentative_g = gscore[current] + 1.4
                else:
                    tentative_g = gscore[current] + 1.0
                if 0 <= neighbor[0]  < self.y_size : 
                    if 0 <= neighbor[1]  < self.x_size :
                        if self.data[neighbor[0]][neighbor[1]] == 1:
                            continue
                    else:
                        continue #array bound x
                else:
                    continue # array bound y

                if neighbor in close_set and tentative_g >= gscore.get(neighbor, 0):
                    continue
                if tentative_g < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g
                    heapq.heappush(oheap, (gscore[neighbor], neighbor))
        print("find no path")
        data = None
        return data
