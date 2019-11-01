import numpy as np

def latex_print(trans_mat):
    for line in trans_mat:
        print(" & ".join([str(num) for num in line]) + "\\\\")
# print(trans_mat)

init_state = np.array([25/150, 20/150, 35/150, 24/150, 46/150])
print(np.average(init_state))
trans_mat = np.array([[0.5, 0.25, 0, 0, 0.25],
[0.25, 0.5, 0.25, 0, 0], 
[0, 0.25, 0.5, 0.25, 0],
[0, 0, 0.25, 0.5, 0.25],
[0.25, 0, 0, 0.25, 0.5]])

for _ in range(100):
    init_state = init_state.dot(trans_mat)

print(init_state)

'''
problem b
'''
init_state = np.array([25, 20, 35, 24, 46])
trans_mat = np.array([[0, 0.5, 0, 0, 0.5],
[0.5, 0, 0.5, 0, 0], 
[0, 0.5, 0, 0.5, 0],
[0, 0, 0.5, 0, 0.5],
[0.5, 0, 0, 0.5, 0]])


# latex_print()

for _ in range(100):
    init_state = init_state.dot(trans_mat)

print(init_state)