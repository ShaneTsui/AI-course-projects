import random
from math import sqrt

import numpy as np

from my_collision.box import Box
from my_collision.vector3 import Vector3
from my_collision.ray import Ray

class RRTRobotPlanner:
    # __slots__ = ['boundary', 'blocks']
    def __init__(self, boundary, blocks):
        self.boundary = boundary
        self.blocks = blocks
        self.boxes = self._create_boxes()

        self.V = set()
        self.E = set()
        self.steer_distance = 0.7
        self.pg = 0.5

    def _create_boxes(self):
        boxes = []
        for row in range(self.blocks.shape[0]):
            boxes.append(Box(Vector3(self.blocks[row, 0], self.blocks[row, 1], self.blocks[row, 2]), Vector3(self.blocks[row, 3], self.blocks[row, 4], self.blocks[row, 5])))
        return boxes

    # TODO: Update this function
    # def is_segment_collision_free(self, x_near, x_new):
    #     return self.is_point_collision_free(x_new)

    def is_segment_collision_free(self, x_near, x_new):
        r = Ray(o=Vector3(x_near[0], x_near[1], x_near[2]), d=Vector3(x_new[0] - x_near[0], x_new[1] - x_near[1], x_new[2] - x_near[2]))
        for box in self.boxes:
            if box.intersect(r):
                return False
        return True and self.is_point_collision_free(x_new)

    def is_point_collision_free(self, node):
        # Check is out of boundary
        if (node[0] < self.boundary[0, 0] or node[0] > self.boundary[0, 3] or \
                node[1] < self.boundary[0, 1] or node[1] > self.boundary[0, 4] or \
                node[2] < self.boundary[0, 2] or node[2] > self.boundary[0, 5]):
            return False

        # Check if the node is inside any block
        for k in range(self.blocks.shape[0]):
            if (node[0] >= self.blocks[k, 0] and node[0] <= self.blocks[k, 3] and \
                    node[1] >= self.blocks[k, 1] and node[1] <= self.blocks[k, 4] and \
                    node[2] >= self.blocks[k, 2] and node[2] <= self.blocks[k, 5]):
                return False
        return True

    def sample_random_point(self):
        x = self.boundary[0, 0] + random.random() * (self.boundary[0, 3] - self.boundary[0, 0])
        y = self.boundary[0, 1] + random.random() * (self.boundary[0, 4] - self.boundary[0, 1])
        z = self.boundary[0, 2] + random.random() * (self.boundary[0, 5] - self.boundary[0, 2])
        return (x, y, z)

    def sample_free(self):
        while True:
            point = self.sample_random_point()
            # Check if the point is collision_old free
            if self.is_point_collision_free(point):
                return point

    def sample_goal_biased(self, goal):
        if random.random() < self.pg:
            return goal
        while True:
            point = self.sample_random_point()
            # Check if the point is collision_old free
            if self.is_point_collision_free(point):
                return point

    @staticmethod
    def euclidean_distance(start: tuple, end: tuple) -> float:
        return sqrt(abs(start[0] - end[0]) ** 2 + abs(start[1] - end[1]) ** 2 + abs(start[2] - end[2]) ** 2)

    def nearest(self, x_rand):
        nn_node, min_dis = None, float('inf')
        for node in self.V:
            dis = self.euclidean_distance(x_rand, node)
            if dis < min_dis:
                nn_node, min_dis = node, dis
        return nn_node

    def steer(self, from_node, to_node):
        distance = self.euclidean_distance(from_node, to_node)
        if distance < self.steer_distance:
            return to_node
        unit_direction_vec = ((to_node[0] - from_node[0]) / distance, \
                              (to_node[1] - from_node[1]) / distance, \
                              (to_node[2] - from_node[2]) / distance)
        return (from_node[0] + self.steer_distance * unit_direction_vec[0], \
                from_node[1] + self.steer_distance * unit_direction_vec[1],
                from_node[2] + self.steer_distance * unit_direction_vec[2])

    def plan(self, start, goal):

        start = tuple(start.tolist())
        goal = tuple(goal.tolist())

        self.V.add(start)

        # Step 1: Samplefree
        # x_rand = self.sample_free()
        x_rand = self.sample_goal_biased(goal)

        # Step 2: Find nearest node in the graph to x_rand
        x_nearest = self.nearest(x_rand)

        # Step 3: Steer
        x_new = self.steer(x_nearest, x_rand)

        # Step 4: Check if the new edge collide with blocks
        if self.is_segment_collision_free(x_nearest, x_new):
            self.V.add(x_new)
            self.E.add((x_nearest, x_new))
            return np.array(x_nearest), np.array(x_new)
        # TODO: If the plan failed, just stay at the same place
        else:
            return np.array(start), np.array(start)
