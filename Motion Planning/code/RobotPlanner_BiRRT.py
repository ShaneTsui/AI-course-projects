import random
from math import sqrt

import numpy as np

from my_collision.box import Box
from my_collision.ray import Ray
from my_collision.vector3 import Vector3


class BiRRTRobotPlanner:
    # __slots__ = ['boundary', 'blocks']
    def __init__(self, boundary, blocks):
        self.boundary = boundary
        self.blocks = blocks
        self.boxes = self._create_boxes()

        self.parent = dict()
        self.Va = set()
        self.Vb = set()
        self.Ea = set()
        self.Eb = set()
        self.steer_distance = 0.8
        self.pg = 0.5

    def init_params(self, start, goal, ax, fig, plt):
        self.start = tuple(start.tolist())
        self.goal = tuple(goal.tolist())
        self.new_goal = self.goal

        self.Va.add(self.start)
        self.Vb.add(self.goal)

        self.ax = ax
        self.fig = fig
        self.plt = plt

    def _create_boxes(self):
        boxes = []
        for row in range(self.blocks.shape[0]):
            boxes.append(Box(Vector3(self.blocks[row, 0], self.blocks[row, 1], self.blocks[row, 2]),
                             Vector3(self.blocks[row, 3], self.blocks[row, 4], self.blocks[row, 5])))
        return boxes

    # Use Bi-RRT for pre-planning. Meanwhile, the agent receives the order to stay at the start point.
    def pre_plan(self):

        # Step 2: Do Bi-RRT search
        # Step 2.1: Samplefree
        # x_rand_a = self.sample_free()
        x_rand_a = self.sample_goal_biased(self.new_goal)

        # Step 2.2: Find nearest node in the graph to x_rand_a
        x_nearest_a = self.nearest(self.Va, self.Ea, x_rand_a)

        # Step 2.3: Steer
        x_new_a = self.steer(x_nearest_a, x_rand_a)

        # Step 3: Check if the new edge collide with blocks
        if self.is_segment_collision_free(x_nearest_a, x_new_a):
            self.Va.add(x_new_a)
            self.Ea.add((x_nearest_a, x_new_a))
            self.Ea.add((x_new_a, x_nearest_a))
            self.parent[x_new_a] = x_nearest_a

            x, y, z = zip(x_nearest_a, x_new_a)
            self.ax.plot(x, y, z, c='r')
            self.fig.canvas.flush_events()
            self.plt.show()

            # Step 4: Update another tree
            # Step 4.1: Find nearest node in the graph to x_rand_a
            x_nearest_b = self.nearest(self.Vb, self.Eb, x_new_a)

            # Step 4.2: Steer towards x_new_a
            x_new_b = self.steer(x_nearest_b, x_rand_a)

            if self.is_segment_collision_free(x_nearest_b, x_new_b):
                self.Vb.add(x_new_b)
                self.Eb.add((x_nearest_b, x_new_b))
                self.Eb.add((x_new_b, x_nearest_b))
                self.parent[x_new_b] = x_nearest_b
                # self.parent[x_nearest_b] = x_new_b

                x, y, z = zip(x_nearest_b, x_new_b)
                self.ax.plot(x, y, z, c='r')
                self.fig.canvas.flush_events()
                self.plt.show()

                if self.is_segment_collision_free(x_new_a, x_new_b) and self.is_segment_collision_free(x_new_b, x_new_a) \
                        and sum((np.array(x_new_a) - np.array(x_new_b)) ** 2) <= 0.1:
                    # self.goal_a, self.goal_b = x_new_a, x_new_b
                    # Ensure start is A
                    if self.start not in self.Va:
                        self.Va, self.Vb = self.Vb, self.Va
                        self.Ea, self.Eb = self.Eb, self.Ea
                        x_new_a, x_new_b = x_new_b, x_new_a

                    if x_new_a == self.goal:
                        self.path = self._find_path(x_new_a)[::-1]
                    elif x_new_b == self.start:
                        self.path = self._find_path(x_new_b)
                    else:
                        self.path = self._find_path(x_new_a)[::-1] + self._find_path(x_new_b)
                    self.smooth()
                    return True

        # Swap two trees to balance the search
        if len(self.Vb) < len(self.Va):
            self.Va, self.Vb = self.Vb, self.Va
            self.Ea, self.Eb = self.Eb, self.Ea
            self.new_goal = self.start if self.new_goal == self.goal else self.goal
        return False

    def _find_path(self, goal):
        path = [goal]
        while path[-1] in self.parent:
            parent = self.parent[path[-1]]
            if parent in path:
                return path
            else:
                path.append(parent)
        return path

    def is_fast(self, p1, p2):
        return sum((np.array(p1) - np.array(p2)) ** 2) > self.steer_distance

    def smooth(self):
        smoothed_path = [self.path[0]]
        cur_point = self.path[0]
        for i in range(2, len(self.path)):
            if not self.is_segment_collision_free(cur_point, self.path[i]) or self.is_fast(cur_point, self.path[i]):
                cur_point = self.path[i - 1]
                smoothed_path.append(cur_point)
        smoothed_path.append(self.path[-1])
        self.path = smoothed_path

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
        return sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2 + (start[2] - end[2]) ** 2)

    @staticmethod
    def manhattan_heuristic(p1: tuple, p2: tuple) -> float:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) + abs(p1[2] - p2[2])

    @staticmethod
    def inf_norm_distance(start: tuple, end: tuple) -> float:
        return max(abs(start[0] - end[0]), abs(start[1] - end[1]), abs(start[2] - end[2]))


    def nearest(self, V, E, x_rand):
        nn_node, min_dis = None, float('inf')
        for node in V:
            dis = self.inf_norm_distance(x_rand, node)
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

    def plan(self):
        for node in self.path:
            yield np.array(node)
