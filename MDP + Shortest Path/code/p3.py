import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict

def cost(x):
    return x.T.dot(x) / 2


def find_best_policy(x, controls, T):

    states, values = defaultdict(list), dict()
    states[0].append(x)
    values[0] = np.array([cost(x)])
    for i in range(T):
        states[i + 1].extend([control.dot(state) for control in controls for state in states[i]])
        values[i + 1] = np.array([cost(state) for state in states[i + 1]]) + np.concatenate((values[i], values[i]))

    policy = []
    T = len(values) - 1
    idx = np.argmin(values[T])
    for t in range(T, 0, -1):
        policy.append(1 if idx < len(values[t]) // 2 else 2)
        idx %= (len(values[t]) // 2)
    return policy[::-1], values[idx]


def plot2d(x, y, labels, time):

    controls, xs, ys = defaultdict(list), defaultdict(list), defaultdict(list)
    for i, label in enumerate(labels):
        controls[label].append(label)
        xs[label].append(x[i])
        ys[label].append(y[i])

    fig, ax = plt.subplots()
    for type in sorted(controls.keys()):
        ax.scatter(xs[type], ys[type], s=[1]*len(xs[type]), label=f'${type}$', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    plt.savefig(f"t{time}", bbox_inches='tight')
    plt.close()


def plot3d(x, y, z, name="v0"):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(np.array(x), np.array(y), np.array(z), rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
    plt.savefig(f"{name}")
    plt.close()


if __name__ == "__main__":
    A = np.array([[0.75, -1], [1, 0.75]])
    B = np.array([[1, 0.5], [0.5, 0.5]])
    policies = []
    X, Y = np.meshgrid(np.arange(-1, 1.01, 0.005), np.arange(-1, 1.01, 0.005))
    values = np.zeros_like(X)

    for row in range(X.shape[0]):
        for col in range(X.shape[1]):
            best_policy, best_value = find_best_policy(np.array([X[row][col], Y[row][col]]), [A, B], T=3)
            policies.append(best_policy)
            values[row][col] = best_value

    plot3d(X, Y, values)

    controls = [p for p in zip(*policies)]
    for time, control in enumerate(controls):
        plot2d(X.reshape(-1), Y.reshape(-1), list(control), time=time)