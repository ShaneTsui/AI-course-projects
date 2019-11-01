"""
==================================
Inverted pendulum animation class
==================================

Adapted from the double pendulum problem animation.
https://matplotlib.org/examples/animation/double_pendulum_animated.html
"""
import math
from time import time
from scipy.stats import multivariate_normal
import numpy as np
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import matplotlib.pyplot as plt

class EnvAnimate:

    '''
    Initialize Inverted Pendulum
    '''

    def _velocity(self, v):
        return int((v - self.vmin) / self.velocity_unit)

    def _theta(self, theta):
        return int((theta - self.theta_min) / self.theta_unit)

    def _control(self, u):
        return int((u - self.umin) / self.control_unit)


    def control_interpolate(self, theta, velocity):

        noise_pdf = multivariate_normal((theta, velocity), self.dt * self.sigma * np.eye(2, 2))
        probs = noise_pdf.pdf(self.states_meshgrid)
        prob_norm = probs / np.sum(probs)
        prob_norm = prob_norm.T
        control_new = np.sum(prob_norm * self.policy)

        theta_new = theta + self.dt * velocity + np.random.normal(0, self.dt) * self.sigma
        velocity_new = velocity + self.dt * (self.a * np.sin(theta) - self.b * velocity + control_new) + np.random.normal(0, self.dt) * self.sigma

        return theta_new, velocity_new, control_new

    def __init__(self):

        # Change this to match your discretization
        self.dt = 0.05
        self.t = np.arange(0.0, 10.0, self.dt)
        self.n_theta = 125   # theta
        self.n_v = 51    # velocity
        self.n_u = 21
        self.theta_max, self.theta_min = math.pi, -math.pi
        self.vmax, self.vmin = 2, -2
        self.umax, self.umin= 2, -2

        self.a = 1
        self.b = 1  # Todo: change b
        self.r = 0.01
        self.k = 1
        self.sigma = 10
        self.gamma = 0.9

        self.thetas = np.linspace(self.theta_min, self.theta_max, self.n_theta + 1)
        self.velocities = np.linspace(self.vmin, self.vmax, self.n_v + 1)
        self.controls = np.linspace(self.umin, self.umax, self.n_u + 1)

        xs, ys = np.meshgrid(self.thetas, self.velocities)
        self.states_meshgrid = np.empty(xs.shape + (2,))
        self.states_meshgrid[:, :, 0], self.states_meshgrid[:, :, 1] = xs, ys

        self.theta_unit = (self.theta_max - self.theta_min) / self.n_theta
        self.velocity_unit = (self.vmax - self.vmin) / self.n_v
        self.control_unit = (self.umax - self.umin) / self.n_u

        self.prob_norm_cache = dict()

        # Plot
        self.observed_states= [(self.theta_min, self.vmin), (0, self.vmin), (self.theta_max, self.vmin), (self.theta_min, self.vmax)]
        self.observed_states_idx = [(0, 0), (64, 0), (125, 0), (0, 51)]


    def simulation(self, theta, velocity):
        thetas, vs, us = [], [], []
        for t in range(self.t.shape[0]):
            theta, velocity, control = self.control_interpolate(theta, velocity)
            thetas.append(theta)
            vs.append(velocity)
            us.append(control)
        return thetas, us


    def _stage_cost(self, theta, control):
        return 1 - np.exp(self.k * np.cos(theta) - self.k) + self.r * (control ** 2) / 2


    def _calc_value(self, theta, velocity, control, V):
        if (theta, velocity, control) in self.prob_norm_cache:
            prob_norm = self.prob_norm_cache[(theta, velocity, control)]
        else:
            theta_new = theta + self.dt * velocity
            velocity_new = velocity + self.dt * (self.a * np.sin(theta) - self.b * velocity + control)
            noise_pdf = multivariate_normal((theta_new, velocity_new), self.dt * self.sigma*np.eye(2, 2))
            probs = noise_pdf.pdf(self.states_meshgrid)
            prob_norm = probs / np.sum(probs)
            prob_norm = prob_norm.T
            self.prob_norm_cache[(theta, velocity, control)] = prob_norm

        costs = np.sum(self.gamma * V * prob_norm)
        stage_cost = self._stage_cost(theta, control)
        return stage_cost + costs


    def value_iteration(self):
        
        # Step 1: Initialize states
        V = np.random.uniform(size=(self.n_theta + 1, self.n_v + 1))
        policy = np.random.uniform(size=(self.n_theta + 1, self.n_v + 1))

        # Step 2: Loop
        threshhold = 1e-5
        iter = 0
        delta = float('inf')

        v_observed = [[] for _ in range(len(self.observed_states))]

        while delta > threshhold:
            iter += 1
            print(iter)
            delta = 0

            for i, (t_idx, v_idx) in enumerate(self.observed_states_idx):
                v_observed[i].append(V[t_idx][v_idx])

            for theta_idx, theta in tqdm(enumerate(self.thetas)):
                for velocity_idx, velocity in enumerate(self.velocities):
                    old_v = V[theta_idx][velocity_idx]
                    min_v, best_u = float('inf'), policy[theta_idx][velocity_idx]
                    for u in self.controls:
                        new_v = self._calc_value(theta, velocity, u, V)
                        if new_v < min_v:
                            min_v, best_u = new_v, u
                    V[theta_idx][velocity_idx], policy[theta_idx][velocity_idx] = min_v, best_u
                    delta = max(delta, abs(old_v - min_v))
            print("Delta = ", delta)

        for i, values in enumerate(v_observed):
            xs = list(range(len(values)))
            plt.plot(xs, values)
            plt.xlabel("Iterations")
            plt.ylabel("Values")
            plt.savefig(f"VI-{i}.png")
            plt.close()

        t = time()
        np.save(f"VI-policy-{t}-{self.sigma}.npy", policy)
        np.save(f"VI-V-{t}-{self.sigma}.npy", V)
        return policy


    def policy_iteration(self):
        # Step 1: Initialize states
        V = np.random.uniform(size=(self.n_theta + 1, self.n_v + 1))
        policy = np.random.uniform(size=(self.n_theta + 1, self.n_v + 1))
        v_observed = [[] for _ in range(len(self.observed_states))]

        is_policy_stable = False
        iter = 0
        while not is_policy_stable:
            iter += 1
            print(iter)

            # Step 2: Policy Evaluation
            print("Policy Evaluation")
            delta = float('inf')
            threshhold = 1e-5
            while delta >= threshhold:
                for i, (t_idx, v_idx) in enumerate(self.observed_states_idx):
                    v_observed[i].append(V[t_idx][v_idx])
                delta = 0
                for theta_idx, theta in tqdm(enumerate(self.thetas)):
                    for velocity_idx, velocity in enumerate(self.velocities):
                        old_v = V[theta_idx][velocity_idx]
                        V[theta_idx][velocity_idx] = self._calc_value(theta, velocity, policy[theta_idx][velocity_idx], V)
                        delta = max(delta, abs(V[theta_idx][velocity_idx] - old_v))
                print("Delta = ", delta)

            # Step 3: Policy Improvement
            print("Policy Improvement")
            is_policy_stable = True
            for theta_idx, theta in tqdm(enumerate(self.thetas)):
                for velocity_idx, velocity in enumerate(self.velocities):
                    old_action = policy[theta_idx][velocity_idx]
                    min_val = float('inf')
                    for u in self.controls:
                        val = self._calc_value(theta, velocity, u, V)
                        if val < min_val:
                            min_val, policy[theta_idx][velocity_idx] = val, u
                    if old_action != policy[theta_idx][velocity_idx]:
                        is_policy_stable = False

        for i, values in enumerate(v_observed):
            xs = list(range(len(values)))
            plt.plot(xs, values)
            plt.xlabel("Iterations")
            plt.ylabel("Values")
            plt.savefig(f"PI-{i}.png")
            plt.close()

        t = time()
        np.save(f"PI-policy-{t}.npy", policy)
        np.save(f"PI-V-{t}.npy", V)
        return policy

    '''
    Provide new rollout theta values to reanimate
    '''
    def new_data(self, theta, controls):
        self.theta = theta
        self.x1 = np.sin(theta)
        self.y1 = np.cos(theta)
        self.u = controls

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111,autoscale_on=False, xlim=(-2,2), ylim=(-2,2))
        self.ax.grid()
        self.ax.axis('equal')
        plt.axis([-2, 2,-2, 2])
        self.line, = self.ax.plot([],[], 'o-', lw=2)
        self.time_template = 'time = %.1fs \nangle = %.2frad\ncontrol = %.2f'
        self.time_text = self.ax.text(0.05, 0.9, '', transform=self.ax.transAxes)


    def init(self, policy=None):
        self.line.set_data([], [])
        self.time_text.set_text('')
        return self.line, self.time_text


    def _update(self, i):
        thisx = [0, self.x1[i]]
        thisy = [0, self.y1[i]]
        self.line.set_data(thisx, thisy)
        self.time_text.set_text(self.time_template % (self.t[i], self.theta[i], self.u[i]))
        return self.line, self.time_text


    def start(self):
        print('Starting Animation')
        # Set up plot to call animate() function periodically
        self.ani = FuncAnimation(self.fig, self._update, frames=range(len(self.x1)), interval=25, blit=True, init_func=self.init, repeat=True)
        plt.show()


    def train(self, policy=None, mode="VI"):
        if policy:
            self.policy = np.load("VI-policy-1560118151.9521945.npy")
            # self.policy = np.load("PI-policy-1560125748.181167.npy")
            # hf = plt.figure()
            # ha = hf.add_subplot(111, projection='3d')
            # X, Y = np.meshgrid(self.thetas, self.velocities)  # `plot_surface` expects `x` and `y` data to be 2D
            # ha.plot_surface(X, Y, self.policy.T, cmap='viridis')
            # ha.set_xlabel("theta")
            # ha.set_ylabel("velocity")
            # ha.set_zlabel("control")
            # plt.show()
            self.V = np.load("VI-V-1560118151.9521945.npy")
        elif mode == "VI":
            self.policy = self.value_iteration()
        elif mode == "PI":
            self.policy = self.policy_iteration()



if __name__ == '__main__':
    animation = EnvAnimate()
    animation.train(policy=True, mode="PI")
    theta, controls= animation.simulation(math.pi, 0)
    animation.new_data(theta, controls)
    animation.start()