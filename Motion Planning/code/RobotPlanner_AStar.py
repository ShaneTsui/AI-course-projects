import heapq
from collections import defaultdict
from math import sqrt
import numpy as np

# from utils import check_collision


class AStarPlanner:
    # __slots__ = ['boundary', 'blocks']

    def __init__(self, boundary, blocks):
        self.boundary = boundary
        self.blocks = blocks

        # Decretize
        self.step = 0.1

        # RTAA*
        self.epsilon = 1
        self.h = dict() # If not in h, use manhattan_heuristic, else update it
        self.state_curr = None
        self.lookahead = 4
        self.parents = dict()

        self.init_params()

    def init_params(self):
        deltas = [-self.step, 0, self.step]
        [dX, dY, dZ] = np.meshgrid(deltas, deltas, deltas)
        dR = list(zip(dX.flatten(), dY.flatten(), dZ.flatten()))
        self.dR_tuple = dR[:13] + dR[14:]  # Remove (0,0,0)
        self.delta_g = {dr: sqrt(dr[0] ** 2 + dr[1] ** 2 + dr[2] ** 2) for dr in self.dR_tuple}

    @staticmethod
    def manhattan_heuristic(start: tuple, end: tuple) -> float:
        return abs(start[0] - end[0]) + abs(start[1] - end[1]) + abs(start[2] - end[2])

    @staticmethod
    def Euclidean_heuristic(start: tuple, end: tuple) -> float:
        return sqrt(abs(start[0] - end[0])**2 + abs(start[1] - end[1])**2 + abs(start[2] - end[2])**2)

    def get_h(self, start_t:tuple, end_t:tuple):
        # return self.Euclidean_heuristic(start_t, end_t) if start_t not in self.h else self.h[start_t]
        return self.epsilon * (self.manhattan_heuristic(start_t, end_t) if start_t not in self.h else self.h[start_t])

    def check_collision(self, segment, block):
        pass

    def check_arrived(self, robotpos:tuple, goal:tuple):
        # Check if the goal is reached
        # TODO: Add input to check_collision
        # return True if not self.check_collision() and sum((robotpos - goal) ** 2) <= 0.1 else False
        return True if sum((np.array([x - y for x, y in zip(robotpos, goal)])) ** 2) <= 0.1 else False

    def expand(self, start:tuple, goal:tuple):
        for i, dr_t in enumerate(self.dR_tuple):
            child_pos = tuple(map(lambda x, y: round(x + y, 2), start, dr_t))

            # Check if in closed
            if child_pos in self.closed:
                break

            # Check if this direction is valid
            if (child_pos[0] < self.boundary[0, 0] or child_pos[0] > self.boundary[0, 3] or \
                    child_pos[1] < self.boundary[0, 1] or child_pos[1] > self.boundary[0, 4] or \
                    child_pos[2] < self.boundary[0, 2] or child_pos[2] > self.boundary[0, 5]):
                continue

            # TODO: Check collision_old
            valid = True
            for k in range(self.blocks.shape[0]):
                if (child_pos[0] >= self.blocks[k, 0] and child_pos[0] <= self.blocks[k, 3] and \
                        child_pos[1] >= self.blocks[k, 1] and child_pos[1] <= self.blocks[k, 4] and \
                        child_pos[2] >= self.blocks[k, 2] and child_pos[2] <= self.blocks[k, 5]):
                    valid = False
                    break
            if not valid:
                break

            # Update valid, non-closed node
            g_sum = self.g[start] + self.delta_g[dr_t]
            if g_sum < self.g[child_pos]:
                self.g[child_pos] = g_sum
                f_new_pos = self.g[child_pos] + self.get_h(child_pos, goal)
                self.min_f_val[child_pos] = f_new_pos
                heapq.heappush(self.open, (f_new_pos, child_pos))
                self.parents[child_pos] = start


    def plan(self, start:np.array, goal:np.array):
        start = tuple(start.tolist())
        goal = tuple(goal.tolist())
        # {node: g_val}
        self.g = defaultdict(lambda: float('inf'))
        self.g[start] = 0
        self.closed = set()

        # [(f_val, node_t)]
        start_f_val = self.get_h(start, goal)
        self.open = [(start_f_val, start)]

        # {node: f_val}
        self.min_f_val = defaultdict(lambda: float('inf'))
        self.min_f_val[start] = start_f_val

        # Step 1: Expand N (loodahead) nodes
        is_arrived = False
        loodahead = self.lookahead
        while self.open and loodahead:
            curr_f, curr_node = heapq.heappop(self.open)
            self.closed.add(curr_node)

            # If arrived, insert goal into closed, then break
            if self.check_arrived(curr_node, goal):
                is_arrived = True
                break

            # screen out the old nodes with updated smaller f_val
            if curr_f == self.min_f_val[curr_node]:
                # Expand 26 surrounding children
                self.expand(curr_node, goal)
                loodahead -= 1

        # If arrived: return goal node
        if is_arrived:
            return np.array(curr_node)

        # If A* search terminates with an empty open list: Fail
        if not self.open:
            print("Fail 1: Goal not found!")
            exit(1)

        # Step 2: Update heuristic
        # Step 2.1: Find the next node to expand
        is_next_found = False
        while self.open:
            curr_f, curr_node = heapq.heappop(self.open)
            # screen out the old nodes with updated shorter distance
            if curr_f == self.min_f_val[curr_node]:
                is_next_found = True
                break

        # If A* search terminates with an empty open list: Fail
        if not is_next_found:
            print("Fail 2: Goal not found!")
            exit(1)

        # Step 2.2: Update heuristic in closed
        f_curr_node = self.g[curr_node] + self.get_h(curr_node, goal)
        for node in self.closed:
            self.h[node] = f_curr_node - self.g[node]

        # Step 3: Move the agent by 1 step
        goal = curr_node
        path = [goal]
        while self.parents[goal] != start:
            goal = self.parents[goal]
            path.append(goal)
        # return np.array(goal)
        return np.array(curr_node) #np.array(goal)

    # def __decretize_3d__(self, x, y, z):
    #     pass
    #
    # # Build graph
    # @staticmethod
    # def __decretize_1d__(start, end, step=0.1):
    #     return np.linspace(start, end, int((end - start)/step + 1))
    #
    # def construct_graph(self):
    #     [Xs, Ys, Zs] = np.meshgrid(self.__decretize_1d__(self.boundary[0, 0], self.boundary[0, 3]), \
    #                                self.__decretize_1d__(self.boundary[0, 1], self.boundary[0, 4]), \
    #                                self.__decretize_1d__(self.boundary[0, 2], self.boundary[0, 5]))
    #
    #     assert Xs.shape == Ys.shape
    #     assert Ys.shape == Zs.shape
    #     self.nodes = np.zeros(shape=(3,) + Xs.shape)
    #     self.nodes[0][:], self.nodes[1][:], self.nodes[2][:] = Xs, Ys, Zs