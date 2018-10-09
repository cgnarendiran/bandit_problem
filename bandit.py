"""
Bandit Algorithms defined.
"""


import numpy as np


class Bandit:
	def __init__(self, n_arms, arm_option = 1):
		# No. of arms 
		self.k = n_arms
		# State Mean values:
		# switch(arm_option):

		# self.q = np.random.random((self.k,))*10.0
		self.q = [1,1,9.5,1,1]
		# State 

	def pull(self, arm):
		return np.random.normal(self.q[arm], 1)




class EpsilonGreedy:
	def __init__(self, epsilon, n_arms):
		self.epsilon = epsilon
		# No. of arms:
		self.k = n_arms
		self.count = None
		self.Q = None

	def initialize(self):
		# Value function:
		self.Q = [0.0] * self.k
		# No. of times each of the arms were pulled:
		self.count = [0] * self.k

	def update(self, chosen_arm, reward):
		a = chosen_arm
		r = reward
		# Update the count:
		self.count[a] += 1
		# Update the value function
		self.Q[a] += (r - self.Q[a])/(self.count[a])

	def select_arm(self):
		z = np.random.random()
		if z< self.epsilon:
			return np.random.randint(0, self.k)
		else:
			return np.argmax(self.Q)

def test_algorithm(bandit, algo, num_simulations = 1000, horizon = 500):
	print("Running for epsilon at: {}".format(algo.epsilon))
	print("Bandit arms q values: {}".format(bandit.q))	

	# record results:
	chosen_arms = np.zeros((num_simulations, horizon))
	rewards = np.zeros((num_simulations, horizon))

	for sim in range(num_simulations):
		# Re-initialize counts and Q values to zero:
		algo.initialize()

		for t in range(horizon):
			chosen_arm = algo.select_arm()
			# print(chosen_arm)
			chosen_arms[sim,t] = chosen_arm

			reward = bandit.pull(chosen_arm)
			rewards[sim,t] = reward
			algo.update(chosen_arm, reward)

	# Average rewards across all sims and compute cumulative rewards
	average_rewards = np.mean(rewards, axis=0)
	cumulative_rewards = np.cumsum(average_rewards)
	# print("Average Rewards {}".format(average_rewards))
	# print("cumulative Rewards {}".format(cumulative_rewards))
	return chosen_arms, average_rewards, cumulative_rewards

	# for epsilon in [0.01, 0.1]:
	# 	for n_arms in range(5,20):


