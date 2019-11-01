from collections import defaultdict
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import cv2
import math
import numpy as np
from tabulate import tabulate

def visualize(directions, filename):
    def draw_direction(x, y, direction, image, color, thickness):
        def draw_arrow(start, end, color, thickness, image, length=20, alpha=127):
            PI = 3.1415926
            if __name__ == '__main__':
                angle = math.atan2(end[1] - start[1], end[0] - start[0])
                cv2.line(image, start, end, color, thickness)

            arrow_x = end[0] + length * math.cos(angle + PI * alpha / 180)
            arrow_y = end[1] + length * math.sin(angle + PI * alpha / 180)
            cv2.line(image, (int(arrow_x), int(arrow_y)), end, color, thickness)

            arrow_x = end[0] + length * math.cos(angle - PI * alpha / 180)
            arrow_y = end[1] + length * math.sin(angle - PI * alpha / 180)
            cv2.line(image, (int(arrow_x), int(arrow_y)), end, color, thickness)

        width, side = 22, 140
        x_center, y_center = x * side + width, y * side + width
        if direction == "West":
            start, end = (x_center + width, y_center), (x_center - width, y_center)
        elif direction == "North":
            start, end = (x_center, width + y_center), (x_center, y_center - width)
        elif direction == "East":
            start, end = (x_center - width, y_center), (x_center + width, y_center)
        else:
            start, end = (x_center, y_center - width), (x_center, y_center + width)

        biasx = biasy = 35
        start = (start[0] + biasx, start[1] + biasx)
        end = (end[0] + biasy, end[1] + biasy)
        draw_arrow(start, end, color, thickness, image)

    color = (255, 0, 0)
    thickness = 10
    arrow_image = mpimg.imread("2_blank_grid.jpg")
    for state in range(1, 26):
        x = (state - 1) % 5
        y = (state - 1) // 5
        draw_direction(x, y, policy[state], arrow_image, color, thickness)
    plt.axis('off')
    plt.imshow(arrow_image)
    plt.savefig(filename)
    plt.close()

def load_motion_model_and_cost(filename="p2.csv"):
    motion_model = defaultdict(dict)
    stage_cost = defaultdict(dict)
    with open(filename, 'r', encoding="utf-8") as f:
        for line in f.readlines()[1:]:
            line = line.strip().split(',')
            xi, u, x_next, cost = int(line[0]), line[1], int(line[2]), int(line[3])
            motion_model[xi][u] = x_next
            stage_cost[xi][u] = cost
    return motion_model, stage_cost


def policy_iteration(states, actions, policy, motion_model, stage_cost, gamma):
    # Step 1: Initialize Values
    V = dict()
    for s in states:
        V[s] = 0.

    is_policy_stable = False
    while not is_policy_stable:

        is_policy_stable = True

        # Step 2: policy evaluation
        delta = float('inf')
        threshhold = 1e-10
        while delta >= threshhold:
            delta = 0
            for s in states:
                old_v = V[s]
                a = policy[s]
                V[s] = stage_cost[s][a] + gamma * V[motion_model[s][a]]
                delta = max(delta, abs(V[s] - old_v))

        # Step 3: policy improvement

        for s in states:
            old_action = policy[s]
            min_val = float('inf')
            for a in actions:
                val = stage_cost[s][a] + gamma * V[motion_model[s][a]]
                if val < min_val:
                    policy[s], min_val = a, val
            if old_action != policy[s]:
                is_policy_stable = False

    return V, policy


def value_iteration(states, actions, policy, motion_model, stage_cost, gamma=0.9):
    # Step 1: Initialize V
    V = dict()
    for s in states:
        V[s] = 0.

    # Step 2: Value iteration
    delta = float('inf')
    threshhold = 1e-10
    while delta >= threshhold:
        delta = 0.
        for s in states:
            old_v = V[s]
            min_val = float('inf')
            for a in actions:
                val = stage_cost[s][a] + gamma * V[motion_model[s][a]]
                if val < min_val:
                    policy[s], min_val = a, val
            V[s] = min_val
            delta = max(delta, abs(V[s] - old_v))
    return V, policy


def q_value_iteration(states, actions, policy, motion_model, stage_cost, gamma=0.9):
    # Step 1: Initialize V and Q
    V = dict()
    for s in states:
        V[s] = 0.

    Q = defaultdict(dict)
    for s in states:
        for a in actions:
            Q[s][a] = 0.

    # Step 2: Q-value iteration
    delta = float('inf')
    threshhold = 1e-10
    while delta >= threshhold:
        delta = 0.
        for s in states:
            old_v = V[s]
            for a in actions:
                Q[s][a] = stage_cost[s][a] + gamma * V[motion_model[s][a]]
                if Q[s][a] < V[s]:
                    policy[s], V[s] = a, Q[s][a]
            delta = max(delta, abs(V[s] - old_v))
    return V, policy



if __name__=="__main__":
    motion_model, stage_cost = load_motion_model_and_cost()

    states = list(range(1, 26))
    actions = ["North", "South", "East", "West"]
    gamma = 0.99

    policy = {s: "North" for s in states}
    V_pi, policy_pi = policy_iteration(states, actions, policy, motion_model, stage_cost, gamma)
    for k, val in V_pi.items():
        print(f"V[{k}] = {val}")
    for s, dir in policy_pi.items():
        print(f"policy[{s}]: {dir}")
    visualize(policy, f"pi_{int(gamma*100)}.png")


    policy = {s: "North" for s in states}
    V_vi, policy_vi = value_iteration(states, actions, policy, motion_model, stage_cost, gamma)
    for k, val in V_vi.items():
        print(f"V[{k}] = {val}")
    for s, dir in policy_vi.items():
        print(f"policy[{s}]: {dir}")
    visualize(policy, f"vi_{int(gamma*100)}.png")

    policy = {s: "North" for s in states}
    V_qi, policy_qi = q_value_iteration(states, actions, policy, motion_model, stage_cost, gamma)
    for k, val in V_qi.items():
        print(f"V[{k}] = {val}")
    for s, dir in policy_qi.items():
        print(f"policy[{s}]: {dir}")
    visualize(policy, f"qi_{int(gamma*100)}.png")

    a = np.zeros(shape=(4, 25))
    a[0] = np.array(list(range(1, 26)))
    a[1] = np.array([V_pi[i] for i in range(1, 26)])
    a[2] = np.array([V_vi[i] for i in range(1, 26)])
    a[3] = np.array([V_qi[i] for i in range(1, 26)])
    a = a.T
    print(tabulate(a, tablefmt="latex", floatfmt=".3f"))