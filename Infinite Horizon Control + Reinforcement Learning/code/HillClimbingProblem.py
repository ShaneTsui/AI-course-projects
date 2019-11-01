import gym
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from tqdm import tqdm


class Planner:

	'''
	Initialization of all necessary variables to generate a policy:
		discretized state space
		control space
		discount factor: gamma
		learning rate: lr
		greedy probability: epsilon
	'''
	def __init__(self, env, gamma, lr, epsilon, max_episodes, greedy_ratio):
		self.env = env
		self.gamma = gamma
		self.lr = lr
		self.epsilon = epsilon
		self.delta_epsilon = epsilon / (max_episodes * greedy_ratio)
		self.max_episodes = max_episodes

		# Discretize state space
		position_max, speed_max = env.observation_space.high[0], env.observation_space.high[1]
		self.position_min, self.speed_min = env.observation_space.low[0], env.observation_space.low[1]
		self.pos_unit, self.speed_unit = 0.05, 0.01
		position_space_num = int((position_max - self.position_min) / self.pos_unit) + 1
		speed_space_num = int((speed_max - self.speed_min) / self.speed_unit) + 1
		action_space_num = env.action_space.n

		self.positions = list(range(position_space_num))
		self.speeds = list(range(speed_space_num))
		self.controls = list(range(action_space_num))

		# Initialize Q table: Q[pos][speed][action] (37, 15, 3)
		self.Q = np.random.uniform(size=(position_space_num, speed_space_num, action_space_num))
		self.Q[position_space_num - 1,:,:] = 0.	# Set terminal state to 0

		self.state_to_plot = [(0., 0.), (-1., 0.05), (0.25, -0.05)]

	'''
	Learn and return a policy via model-free policy iteration.
	'''
	def __call__(self, mc=False, on=True):
		self._td_policy_iter(on)


	def _state(self, state):
		return (int((state[0] - self.position_min) / self.pos_unit), int((state[1] - self.speed_min) / self.speed_unit))


	def _epsilon_greedy(self, state, epsilon):
		if np.random.random() <= epsilon:
			return np.random.choice(range(self.env.action_space.n))
		else:
			return np.argmin(self.Q[state[0], state[1]])


	def policy(self, state):
		state = self._state(state)
		return np.argmin(self.Q[state[0], state[1], :])


	def _sarsa(self):

		# Plot
		q_plot = np.zeros(shape=(3, env.action_space.n, self.max_episodes))
		state_to_plot = [self._state(s) for s in self.state_to_plot]

		for episode in tqdm(range(self.max_episodes)):

			# Plot
			for i, s in enumerate(state_to_plot):
				for action in range(env.action_space.n):
					q_plot[i, action, episode] = self.Q[s[0], s[1], action]

			state_prev = self._state(self.env.reset())
			is_finished = False
			reward_total = 0.
			control = self._epsilon_greedy(state_prev, self.epsilon)

			while not is_finished:

				# Move 1 step further
				state_new, reward, is_finished, _ = env.step(control)
				stage_cost = -reward
				state_new = self._state(state_new)
				reward_total += stage_cost

				control_new = self._epsilon_greedy(state_new, self.epsilon)

				if not is_finished or state_new[0] < 50:
					self.Q[state_prev[0], state_prev[1], control] += lr * (stage_cost \
								+ gamma * self.Q[state_new[0], state_new[1], control_new] \
								- self.Q[state_prev[0], state_prev[1], control])
				else:  # terminates
					self.Q[state_prev[0], state_prev[1], control] = stage_cost

				state_prev, control = state_new, control_new

			if self.epsilon > 0:
				self.epsilon -= self.delta_epsilon

		for action in range(env.action_space.n):
			for i, s in enumerate(self.state_to_plot):
				plt.plot(list(range(self.max_episodes)), q_plot[i, action, :])
				plt.savefig(f"sarsa_{int(s[0] * 100)}_{int(s[1] * 100)}_{action}.png")
				plt.close()


	def _q_learning(self):
		# Plot
		q_plot = np.zeros(shape=(3, env.action_space.n, self.max_episodes))
		state_to_plot = [self._state(s) for s in self.state_to_plot]

		for episode in tqdm(range(self.max_episodes)):

			# Plot
			for i, s in enumerate(state_to_plot):
				for action in range(env.action_space.n):
					q_plot[i, action, episode] = self.Q[s[0], s[1], action]

			state_prev = self._state(self.env.reset())
			is_finished = False
			reward_total = 0.

			while not is_finished:

				control = self._epsilon_greedy(state_prev, self.epsilon)

				# Move 1 step further
				state_new, reward, is_finished, _ = env.step(control)
				state_new = self._state(state_new)
				stage_cost = - reward
				reward_total += stage_cost

				if not is_finished or state_new[0] < 50:
					self.Q[state_prev[0], state_prev[1], control] += lr * (stage_cost + gamma * np.min(self.Q[state_new[0], state_new[1], :]) \
															 - self.Q[state_prev[0], state_prev[1], control])
				else: # terminates
					self.Q[state_prev[0], state_prev[1], control] = stage_cost

				state_prev = state_new

			if self.epsilon > 0:
				self.epsilon -= self.delta_epsilon

		for action in range(env.action_space.n):
			for i, s in enumerate(self.state_to_plot):
				plt.plot(list(range(self.max_episodes)), q_plot[i, action, :])
				plt.savefig(f"ql_{int(s[0] * 100)}_{int(s[1] * 100)}_{action}.png")
				plt.close()

	def show_policy(self, name):
		from matplotlib import pyplot as plt
		import numpy as np
		from mpl_toolkits.mplot3d import Axes3D
		figure = plt.figure()
		ax = Axes3D(figure)
		ax.set_xlabel("Position")
		ax.set_ylabel("Speed")
		ax.set_zlabel("Action")
		position = np.arange(-120, 65, 5) / 100
		speed = np.arange(-7, 8) / 100
		position, speed = np.meshgrid(position, speed)
		action = np.zeros_like(position)
		for i in range(position.shape[0]):
			for j in range(position.shape[1]):
				action[i][j] = self.policy((position[i][j], speed[i][j]))
		# ax.plot_surface(position, speed, action, rstride=1, cstride=1, cmap='rainbow')
		ax.contour(position, speed, action, cmap="RdBu_r")
		plt.show()
		# plt.savefig(f"policy_{name}_scatter.png")
		plt.close()

		# res = []
		# for position in range(-120, 65, 5):
		# 	for speed in range(-7, 8):
		# 		p, s = position/100, speed/100
		# 		action = self.policy((p, s))
		# 		res.append([p, s, action])
		# print(tabulate(res, tablefmt="latex"))

	'''
	TO BE IMPLEMENT
	TD Policy Iteration
	Flags: on : on vs. off policy learning
	Returns: policy that minimizes Q wrt to controls
	'''
	def _td_policy_iter(self, on=True):
		if on:
			self._sarsa()
			self.show_policy("sarsa")
		else:
			self._q_learning()
			self.show_policy("ql")


	'''
	Sample trajectory based on a policy
	'''
	def rollout(self, env, policy=None, render=False):
		traj = []
		t = 0
		done = False
		c_state = env.reset()
		if policy is None:
			while not done or t < 200:
				action = env.action_space.sample()
				if render:
					env.render()
				n_state, reward, done, _ = env.step(action)
				traj.append((c_state, action, reward))
				c_state = n_state
				t += 1

			env.close()
			return traj

		else:
			while not done or t < 200:
				action = policy(c_state)
				if render:
					env.render()

				n_state, reward, done, _ = env.step(action)
				traj.append((c_state, action, reward))
				c_state = n_state
				t += 1

			env.close()
			return traj


if __name__ == '__main__':

	env = gym.make('MountainCar-v0')

	# Hyper-parameters
	gamma = 0.9
	lr = 0.3
	epsilon = 0.8
	max_episodes = 5000
	greedy_ratio = 0.7

	planner = Planner(env, gamma, lr, epsilon, max_episodes, greedy_ratio)

	planner(mc=False, on=True)
	traj = planner.rollout(env, policy=planner.policy, render=True)
	print(traj)

