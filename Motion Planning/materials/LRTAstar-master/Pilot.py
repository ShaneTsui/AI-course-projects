from queue import PriorityQueue
from Config import EnvironmentDef
from State import State


def main():
    env = EnvironmentDef("Environment1.json")
    # env.print_boxes()
    # env.draw_env()
    print(env)
    obstacle_list = set()  #Known obstacles for the robot
    traversal_path = []

    def a_star(present_state):
        open_set_pq, open_set, closed_set = PriorityQueue(), {}, {}

        goal_state = State(env.goal_state, env.resolution, obstacle_list)
        initial_state = State(present_state, env.resolution, obstacle_list, GOAL_STATE=goal_state)
        open_set_pq.put(initial_state)
        open_set[initial_state.position] = initial_state
        state_examinations, state_generated = 0, 1
        while True:
            if not open_set_pq.empty():
                current_state = open_set_pq.get()
                state_examinations += 1
                open_set.pop(current_state.position)
                closed_set[current_state.position] = current_state
            else:
                print('No solution')
                exit()
            if current_state.position == goal_state.position:
                return current_state, state_examinations, state_generated
            else:
                for x in current_state.successor():
                    if x.position in closed_set or x.position in obstacle_list:
                        continue
                    g_cost = current_state.g + 1      #STEP_COST --> 1
                    if x.position not in open_set or g_cost < open_set[x.position].g:
                        x.g = g_cost
                        x.parent = current_state
                        if x.position not in open_set:
                            state_generated += 1
                            open_set[x.position] = x
                            open_set_pq.put(x)

    def lrta_star(present_state, total_restarts_lrta_star):
        print("lrta_star restart number: ", total_restarts_lrta_star)

        def tree_traversal():
            current_state, state_examinations, state_generated = a_star(present_state)
            path = []
            while current_state.parent is not None:
                path.append(current_state)
                current_state = current_state.parent
            path.reverse()
            return path, state_generated, state_examinations

        path, state_generated, state_examinations = tree_traversal()

        def go_to_next_position(i):
            # print("i: %d path %s" %(i, path))
            current_position_traversed = path[i]
            x, y = current_position_traversed.position[0], current_position_traversed.position[1]
            print(x, y, env.is_point_inside_box(x, y))
            if env.is_point_inside_box(x, y) or current_position_traversed.position == tuple(env.goal_state):
                return current_position_traversed
            else:
                traversal_path.append(current_position_traversed)
                return go_to_next_position(i + 1)

        current_position = go_to_next_position(0)
        if current_position.position == tuple(env.goal_state):
            print(traversal_path)
            env.draw_env(traversal_path)
            # print('Total steps were %d, generated %d and expanded %d ' % (len(path), state_generated, state_examinations))

        else:
            obstacle_list.add(current_position.position)
            print("obstacle list is ", obstacle_list)
            lrta_star(current_position.parent.position, total_restarts_lrta_star + 1)

    lrta_star(env.initial_state, 0)
    print(env.internal_is_point_inside(16, 13))

if __name__ == '__main__':
    main()