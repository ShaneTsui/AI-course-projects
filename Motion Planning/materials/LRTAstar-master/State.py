STEP_COST = 1


class State:

    def __init__(self, position, resolution, obstacle_list, GOAL_STATE=None, operation='-', parent=None):
        self.position = tuple(position)
        self.GOAL_STATE = GOAL_STATE
        self.obstacle_list = obstacle_list
        self.parent = parent
        self.g, self.h = 0, 0
        self.successor_list = []
        self.operation = operation
        self.resolution = resolution
        if GOAL_STATE is not None:
            self.heuristics()

    def heuristics(self):
        # h is based on manhattan distance from current state to goal state
        self.h += abs(self.position[0]-self.GOAL_STATE.position[0]) + \
                  abs(self.position[1]-self.GOAL_STATE.position[1])

    def get_f(self):
        return self.g + self.h

    def successor(self):
        #Generates possible childs by moving empty cell to a maximum of 4 possibilities
        i_row, y_col = self.position[0], self.position[1]
        if i_row - 1 >= 0:     #left
            position = tuple((i_row - 1, y_col))
            # print('left' , position)
            if position not in self.obstacle_list:
                self.successor_list.append(State(position, self.resolution, self.obstacle_list, self.GOAL_STATE, operation='Left'))
        if i_row + 1 < self.resolution:    #right
            position = tuple((i_row + 1, y_col))
            # print('right', position)
            if position not in self.obstacle_list:
                self.successor_list.append(State(position, self.resolution, self.obstacle_list, self.GOAL_STATE, operation='Right'))
        if y_col - 1 >= 0:    #down
            position = tuple((i_row, y_col - 1))
            # print('up' , position)
            if position not in self.obstacle_list:
                self.successor_list.append(State(position, self.resolution, self.obstacle_list, self.GOAL_STATE, operation='Down'))
        if y_col + 1 < self.resolution:    #up
            position = tuple((i_row, y_col + 1))
            # print('down', position)
            if position not in self.obstacle_list:
                self.successor_list.append(State(position, self.resolution, self.obstacle_list, self.GOAL_STATE, operation='Up'))
        # if len(self.successor_list) < 4:
        #     print("Generated %d successors." % len(self.successor_list))
        return self.successor_list

    def __lt__(self, other):
        return self.get_f() < other.get_f()

    def __repr__(self):
        return 'State : %s, h : %d operation : %s\n' %(self.position, self.h, self.operation)