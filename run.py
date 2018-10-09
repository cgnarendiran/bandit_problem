
from utils import plot_algorithm, compare_algorithms
from bandit import Bandit
import numpy as np

if __name__ == "__main__":
	# Choose the number of arms:
	n_arms = 5
	# Epsilon value set:
	epsilon_values = [0.1]
	# Plot the average reward:
	plot_algorithm(alg_name = "epsilon-Greedy",  n_arms = n_arms,
        hyper_params=epsilon_values, num_simulations=1000, horizon=1000, label="eps",
        fig_size=(18, 6))
