# import numpy as np

# A = np.mat([[1,2], [3,4]])
# b = np.mat([[1], [2]])
# x = np.linalg.solve(A,b)
# print(x)

import numpy as np
from collections import defaultdict

# Step 1: Read files
'''
Rewards = 81-d column array
Probs = dict{action : {s:[(s1', p1),...]}}
'''

REWARD_FILE = 'rewards.txt'
P_FILE = 'prob_a.txt'

def read_rewards(fname):
    with open(REWARD_FILE) as f:
        return np.mat([int(line.strip()) for line in f]).T

def read_probs(fname):
    probs = dict()
    for action in range(1, 5):
        probs[action] = defaultdict(list)
        name, ext = fname.split('.')
        filename = name + str(action) + '.' + ext
        with open(filename) as f:
            for line in f:
                content = line.split()
                s_cur, s_next, p = int(content[0]), int(content[1]), float(content[2])
                probs[action][s_cur].append((s_next, p))
    return probs

rewards = read_rewards(REWARD_FILE)
probs = read_probs(P_FILE)

# Step 2: Initialize variables
gamma = 0.99
states = range(1, 82)
actions = range(1, 5)
policy = {state: np.random.randint(low = 1, high = 5) for state in states}



# Step 3: Start policy iteration

def evaluate_values():
    M = np.eye(len(states))
    for state in states:
        action = policy[state]
        for s_next, p in probs[action][state]:
            M[state - 1, s_next - 1] -= gamma * p
    return np.linalg.solve(M, rewards)

# values = np.mat() : MINUS 1
# Evaluate Values
def q_sa(state, action, values):
    reward = 0
    for s_next, p in probs[action][state]:
        reward += p * values[s_next - 1]
    return rewards[state - 1] + gamma * reward

# Greedy update policy
def update_policy(values):
    is_updated = False
    policy_new = {state: None for state in states}
    for state in states:
        q_max, action_best = float('-inf'), None
        for action in actions:
            q_sa_value = q_sa(state, action, values)
            if q_max < q_sa_value:
                q_max, action_best = q_sa_value, action
        policy_new[state] = action_best
        if action_best != policy[state]:
            is_updated = True
    return is_updated, policy_new

is_updated = True
iter = 0
while is_updated:
    print(iter)
    values = evaluate_values()
    is_updated, policy = update_policy(values)
    iter += 1

best_value = evaluate_values()
np.savetxt('bestvalue.txt', best_value.reshape((9, 9)).T, fmt='%1.2e')
best_policy = np.array([action for _, action in sorted(list(policy.items()))]).reshape((9, 9)).T
np.savetxt('bestpolicy.txt', best_policy, fmt='%1.0e')
# print(best_policy)