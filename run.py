
from utils import plot_algorithm, compare_algorithms
import numpy as np

if __name__ == "__main__":
	# Choose the number of arms:
	n_arms = 5
	# Epsilon value set:
	epsilon_values = [0.05,0.1,0.2,0.3]
	# Plot the average reward:
	compare_algorithms(algorithms = ["epsilon-Greedy","optimistic-Greedy", "UCB"],  n_arms = n_arms,
        hyper_params=epsilon_values, num_simulations=1000, horizon=500, label="eps",
        fig_size=(18, 6))
