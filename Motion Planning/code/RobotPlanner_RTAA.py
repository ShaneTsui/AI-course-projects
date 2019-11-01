# import heapq
from queue import PriorityQueue
from collections import defaultdict
from math import sqrt
import numpy as np

# from utils import check_collision
from my_collision.box import Box
from my_collision.ray import Ray
from my_collision.vector3 import Vector3


class RTAAPlanner:
    # __slots__ = ['boundary', 'blocks']

    def __init__(self, boundary, blocks):
        self.boundary = boundary
        self.blocks = blocks
        self.boxes = self._create_boxes()

        # Decretize
        self.step = 0.4

        # RTAA*
        self.epsilon = 2
        self.h = dict() # If not in h, use euclidean_heuristic, else update it
        self.lookahead = 15
        self.parents = dict()

        self.decretize()


    def decretize(self, step=None):
        if step:
            self.step = step
        # Init parameters
        deltas = [-self.step, 0, self.step]
        [dX, dY, dZ] = np.meshgrid(deltas, deltas, deltas)
        dR = list(zip(dX.flatten(), dY.flatten(), dZ.flatten()))
        self.dR_tuple = dR[:13] + dR[14:]  # Remove (0,0,0)
        self.delta_g = {dr: sqrt(dr[0] ** 2 + dr[1] ** 2 + dr[2] ** 2) for dr in self.dR_tuple}

    @staticmethod
    def manhattan_heuristic(p1: tuple, p2: tuple) -> float:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) + abs(p1[2] - p2[2])

    @staticmethod
    def inf_norm_distance(start: tuple, end: tuple) -> float:
        return max(abs(start[0] - end[0]), abs(start[1] - end[1]), abs(start[2] - end[2]))

    @staticmethod
    def euclidean_heuristic(p1: tuple, p2: tuple) -> float:
        return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

    def get_h(self, point:tuple):
        # return self.euclidean_heuristic(start_t, end_t) if start_t not in self.h else self.h[start_t]
        # return self.epsilon * (self.manhattan_heuristic(point, self.goal) if point not in self.h else self.h[point])
        # return self.epsilon * (self.inf_norm_distance(point, self.goal) if point not in self.h else self.h[point])
        return self.epsilon * (self.euclidean_heuristic(point, self.goal) if point not in self.h else self.h[point])

    def is_segment_collision_free(self, x_near, x_new):
        r = Ray(o=Vector3(x_near[0], x_near[1], x_near[2]),
                d=Vector3(x_new[0] - x_near[0], x_new[1] - x_near[1], x_new[2] - x_near[2]))
        for box in self.boxes:
            if box.intersect(r):
                return False
        return True and self.is_point_collision_free(x_new)

    def is_point_collision_free(self, node):
        # Check is out of boundary
        if (node[0] <= self.boundary[0, 0] or node[0] >= self.boundary[0, 3] or \
                node[1] <= self.boundary[0, 1] or node[1] >= self.boundary[0, 4] or \
                node[2] <= self.boundary[0, 2] or node[2] >= self.boundary[0, 5]):
            return False

        # Check if the node is inside any block
        for k in range(self.blocks.shape[0]):
            if (node[0] >= self.blocks[k, 0] and node[0] <= self.blocks[k, 3] and \
                    node[1] >= self.blocks[k, 1] and node[1] <= self.blocks[k, 4] and \
                    node[2] >= self.blocks[k, 2] and node[2] <= self.blocks[k, 5]):
                return False
        return True

    def is_arrived(self, robotpos:tuple, goal:tuple):
        # Check if the goal is reached
        # TODO: Add input to check_collision
        # return True if not self.check_collision() and sum((robotpos - goal) ** 2) <= 0.1 else False
        return True if sum((np.array(robotpos) - np.array(goal)) ** 2) <= 0.1 else False

    def is_out_of_boundary(self, point):
        if (point[0] < self.boundary[0, 0] or point[0] > self.boundary[0, 3] or \
                point[1] < self.boundary[0, 1] or point[1] > self.boundary[0, 4] or \
                point[2] < self.boundary[0, 2] or point[2] > self.boundary[0, 5]):
            return True
        else:
            return False

    def plan(self, start:np.array, goal:np.array):

        # Step 0: Initialize the parameters
        start = tuple(start.tolist())
        goal = tuple(goal.tolist())
        closed = set()

        # Check if close:
        if self.euclidean_heuristic(start, goal) < 1:
            self.decretize(0.1)

        open_pq = PriorityQueue()
        open_pq.put((self.get_h(start), start))
        min_g_value = defaultdict(lambda: float('inf'))
        min_g_value[start] = 0
        f_values = dict()
        f_values[start] = self.get_h(start)

        # Step 1: Expand N (nodes) ahead
        # Check if the open_pq is empty
        lookahead = self.lookahead

        while not open_pq.empty() and lookahead:

            # Step 1: Take out the node with smallest f value
            _, current = open_pq.get()
            closed.add(current)

            # Step 2: Check if goal arrived
            if self.is_arrived(current, goal):
                break

            # Step 3: Expand the neighbours
            for i, dr_t in enumerate(self.dR_tuple):
                next_node = tuple(map(lambda x, y: round(x + y, 2), current, dr_t))

                # Check if this direction is valid
                if next_node in closed or self.is_out_of_boundary(next_node) or not self.is_segment_collision_free(current, next_node):
                    continue

                # for next_node in graph.neighbors(current):
                new_g_value = min_g_value[current] + self.delta_g[dr_t]#graph.cost(current, next_node)

                # Update: new node OR smaller cost
                if new_g_value < min_g_value[next_node]:
                    min_g_value[next_node] = new_g_value
                    f_value = new_g_value + self.get_h(next_node)
                    f_values[next_node] = f_value
                    open_pq.put((f_value, next_node))
                    self.parents[next_node] = current

            lookahead -= 1

        if open_pq.empty():
            print("Fail 1: Goal not found!")
            exit(1)

        _, next_to_expand = open_pq.get()

        # Step 2.2: Update heuristic in closed
        f_next_node = min_g_value[next_to_expand] + self.get_h(next_to_expand)
        for node in closed:
            self.h[node] = f_next_node - min_g_value[node]

        # Step 3: Move the agent by 1 step
        target = next_to_expand
        path = [target]
        while self.parents[target] != start:
            target = self.parents[target]
            path.append(target)
        return np.array(path[-1])

    def _create_boxes(self):
        boxes = []
        for row in range(self.blocks.shape[0]):
            boxes.append(Box(Vector3(self.blocks[row, 0], self.blocks[row, 1], self.blocks[row, 2]),
                             Vector3(self.blocks[row, 3], self.blocks[row, 4], self.blocks[row, 5])))
        return boxes

    def set_goal(self, goal):
        self.goal = goal