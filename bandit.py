"""
Bandit Algorithms defined.
"""


import numpy as np


class Bandit:
	def __init__(self, n_arms, arm_option = 1):
		# No. of arms 
		self.k = n_arms
		# State Mean values:
		# self.q = np.random.random((self.k,))*10.0
		# self.q = [1.5,1.5,9.5,1.5,1.5]
		self.q = [4.0,4.5,5.0,5.5,6.0]
		# State 

	def pull(self, arm):
		return np.random.normal(self.q[arm], 1)




class EpsilonGreedy:
	def __init__(self, epsilon, n_arms):
		self.epsilon = epsilon
		# No. of arms:
		self.k = n_arms
		# No. of times the arm is pulled
		self.count = None
		# q estimate of that particular arm
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
		# Update the estimate of the value function
		self.Q[a] += (r - self.Q[a])/(self.count[a])

	def select_arm(self):
		z = np.random.random()
		if z< self.epsilon:
			return np.random.randint(0, self.k)
		else:
			return np.argmax(self.Q)




class OptimisticGreedy:
	def __init__(self, n_arms):
		# No. of arms:
		self.k = n_arms
		# No. of times the arm is pulled
		self.count = None
		# q estimate of that particular arm
		self.Q = None

	def initialize(self):
		# Value function:
		self.Q = [10.0] * self.k
		# No. of times each of the arms were pulled:
		self.count = [0] * self.k

	def update(self, chosen_arm, reward):
		a = chosen_arm
		r = reward
		# Update the count:
		self.count[a] += 1
		# Update the estimate of the value function
		self.Q[a] += (r - self.Q[a])/(self.count[a])

	def select_arm(self):
		return np.argmax(self.Q)




class UCB:

	"""Implementing `UCB` algorithm.
	Parameters
	----------
	counts : list or array-like
	    number of times each arm was played, shape (num_arms,).
	values : list or array-like
	    estimated value (mean) of rewards of each arm, shape (num_arms,).
	Attributes
	----------
	select_arm : int
	    select the arm based on the knowledge we know about each one. All arms
	    will be rescaled based on how many times they were selected to avoid
	    neglecting arms that are good overall but the algorithm has had a
	    negative initial interactions.
	"""
	def __init__(self, n_arms):
		# No. of arms:
		self.k = n_arms
		# No. of times the arm is pulled
		self.count = None
		# q estimate of that particular arm
		self.Q = None

	def initialize(self):
		"""Initialize counts and values array with zeros."""
		# Value function:
		self.Q = [10.0] * self.k
		# No. of times each of the arms were pulled:
		self.count = [0] * self.k
	
	def select_arm(self):
		# Make sure to visit each arm at least once at the beginning
		for arm in range(self.k):
			if self.count[arm] == 0:
				return arm

		# Compute estimated value using original values and bonus
		ucb_values = np.zeros(self.k)
		n = np.sum(self.count)
		for arm in range(self.k):
			# Rescale based on total counts and arm_specific count
			bonus = np.sqrt((np.log(n)) / (2 * self.count[arm]))
			ucb_values[arm] = self.Q[arm] + bonus

		return np.argmax(ucb_values)

	def update(self, chosen_arm, reward):
		"""Update counts and estimated value of rewards for the chosen arm."""
		# Increment counts
		a = chosen_arm
		r = reward
		# Update the count:
		self.count[a] += 1
		# Update the estimate of the value function
		self.Q[a] += (r - self.Q[a])/(self.count[a])



def test_algorithm(bandit, algo, num_simulations = 1000, horizon = 500):

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


