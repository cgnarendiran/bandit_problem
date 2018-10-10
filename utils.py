"""
Plots Bandit Algorithms performance.
"""


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from bandit import EpsilonGreedy, OptimisticGreedy, UCB, Bandit, test_algorithm
# UCB


# % matplotlib inline
plt.style.use("fivethirtyeight")
sns.set_context("notebook")


ALGORITHMS = {
    "epsilon-Greedy": EpsilonGreedy,
    "optimistic-Greedy": OptimisticGreedy,
    "UCB": UCB
}


def plot_algorithm(
        alg_name="epsilon-Greedy", n_arms = 10,
        hyper_params=None, num_simulations=1000, horizon=500, label=None,
        fig_size=(18, 6)):
    # Make a bandit:
    bandit = Bandit(n_arms)
    # Find the best arm:
    best_arm_index = np.argmax(bandit.q)
    # Check if the algorithm doesn't have hyperparameter
    if hyper_params is None:
        # Create the algorithm
        algo = ALGORITHMS[alg_name](n_arms)
        # Run the algorithm:
        chosen_arms, average_rewards, cum_rewards = test_algorithm(bandit, algo, num_simulations, horizon)

        average_probs = np.where(chosen_arms == best_arm_index, 1, 0).sum(axis=0) / num_simulations

        # Plot the 3 metrics of the algorithm
        fig, axes = plt.subplots(1, 3, figsize=fig_size)
        axes[0].plot(average_probs)
        axes[0].set_xlabel("Time", fontsize=14)
        axes[0].set_ylabel("Probability of Selecting Best Arm", fontsize=14)
        axes[0].set_title("Accuray of {} alg.".format(alg_name), y=1.05, fontsize=16)
        axes[0].set_ylim([0, 1.05])
        axes[1].plot(average_rewards)
        axes[1].set_xlabel("Time", fontsize=14)
        axes[1].set_ylabel("Average Reward", fontsize=14)
        axes[1].set_title("Avg. Rewards of {} alg.".format(alg_name), y=1.05, fontsize=16)
        axes[1].set_ylim([0, 10.05])
        axes[2].plot(cum_rewards)
        axes[2].set_xlabel("Time", fontsize=14)
        axes[2].set_ylabel("Cumulative Rewards of Chosen Arm", fontsize=14)
        axes[2].set_title("Cumulative Rewards of {} alg.".format(alg_name), y=1.05, fontsize=16)
        plt.tight_layout()

    else:
        fig, axes = plt.subplots(1, 3, figsize=fig_size)
        for hyper_param in hyper_params:
            # Create the algorithm
            algo = ALGORITHMS[alg_name](hyper_param, n_arms)
            # Run the algorithm:
            chosen_arms, average_rewards, cum_rewards = test_algorithm(bandit, algo, num_simulations, horizon)

            average_probs = np.where(chosen_arms == best_arm_index, 1, 0).sum(axis=0) / num_simulations

            # Plot the 3 metrics of the algorithm
            axes[0].plot(average_probs, label="{} = {}".format(label,hyper_param))
            axes[0].set_xlabel("Time", fontsize=14)
            axes[0].set_ylabel("Probability of Selecting Best Arm", fontsize=14)
            axes[0].set_title("Accuray of {} alg.".format(alg_name), y=1.05, fontsize=16)
            axes[0].legend()
            axes[0].set_ylim([0, 1.05])
            axes[1].plot(average_rewards, label="{} = {}".format(label,hyper_param))
            axes[1].set_xlabel("Time", fontsize=14)
            axes[1].set_ylabel("Average Reward", fontsize=14)
            axes[1].set_title("Avg. Rewards of {} alg.".format(alg_name), y=1.05, fontsize=16)
            axes[1].legend()
            axes[1].set_ylim([0, 10.05])
            axes[2].plot(cum_rewards, label="{} = {}".format(label,hyper_param))
            axes[2].set_xlabel("Time", fontsize=14)
            axes[2].set_ylabel("Cumulative Rewards of Chosen Arm", fontsize=14)
            axes[2].set_title("Cumulative Rewards of {} alg.".format(alg_name), y=1.05, fontsize=16)
            axes[2].legend(loc="lower right")
            plt.tight_layout()
        plt.show()


def compare_algorithms(
        algorithms=None, n_arms=None, 
        hyper_params=None, num_simulations=1000, horizon=100, label=None,
        fig_size=(18, 6)):

    # Make a bandit:
    bandit = Bandit(n_arms)
    print("Bandit arms q values: {}".format(bandit.q))
    # Find the best arm:
    best_arm_index = np.argmax(bandit.q)
    # Check if the algorithm doesn't have hyperparameter
    fig, axes = plt.subplots(1, 3, figsize=fig_size)
    # Loop over all algorithms
    for algorithm in algorithms:

        if (algorithm == "epsilon-Greedy"):
            for hyper_param in hyper_params:
                # Create the algorithm
                algo = ALGORITHMS[algorithm](hyper_param, n_arms)
                print("Running {} for epsilon at: {}".format(algorithm, algo.epsilon))
                # Run the algorithm:
                chosen_arms, average_rewards, cum_rewards = test_algorithm(bandit, algo, num_simulations, horizon)

                average_probs = np.where(chosen_arms == best_arm_index, 1, 0).sum(axis=0) / num_simulations

                # Plot the 3 metrics of the algorithm
                axes[0].plot(average_probs, label="{} = {}".format(label,hyper_param))
                axes[0].set_xlabel("Time", fontsize=12)
                axes[0].set_ylabel("Probability of Selecting Best Arm", fontsize=12)
                axes[0].set_title("Accuray of {} alg.".format(algorithm), y=1.05, fontsize=14)
                axes[0].legend(loc="lower right")
                axes[0].set_ylim([0, 1.05])
                axes[1].plot(average_rewards, label="{} = {}".format(label,hyper_param))
                axes[1].set_xlabel("Time", fontsize=12)
                axes[1].set_ylabel("Average Reward", fontsize=14)
                axes[1].set_title("Avg. Rewards of {} alg.".format(algorithm), y=1.05, fontsize=14)
                axes[1].legend(loc="lower right")
                axes[1].set_ylim([0, 10.05])
                axes[2].plot(cum_rewards, label="{} = {}".format(label,hyper_param))
                axes[2].set_xlabel("Time", fontsize=12)
                axes[2].set_ylabel("Cumulative Rewards of Chosen Arm", fontsize=12)
                axes[2].set_title("Cumulative Rewards of {} alg.".format(algorithm), y=1.05, fontsize=14)
                axes[2].legend(loc="lower right")
                plt.tight_layout()
        else:
            # Create the algorithm
            algo = ALGORITHMS[algorithm](n_arms)
            print("Running {} ".format(algorithm))
            # Run the algorithm
            chosen_arms, average_rewards, cum_rewards = test_algorithm(bandit, algo, num_simulations, horizon)
            average_probs = np.where(chosen_arms == best_arm_index, 1, 0).sum(axis=0) / num_simulations

            # Plot the 3 metrics
            axes[0].plot(average_probs, label=algorithm)
            axes[0].set_xlabel("Time", fontsize=12)
            axes[0].set_ylabel("Probability of Selecting Best Arm", fontsize=12)
            axes[0].set_title("Accuray of Different Algorithms", y=1.05, fontsize=14)
            axes[0].set_ylim([0, 1.05])
            axes[0].legend(loc="lower right")
            axes[1].plot(average_rewards, label=algorithm)
            axes[1].set_xlabel("Time", fontsize=12)
            axes[1].set_ylabel("Average Reward", fontsize=12)
            axes[1].set_title("Average Rewards of Different Algorithms", y=1.05, fontsize=14)
            axes[1].set_ylim([0, 10.05])
            axes[1].legend(loc="lower right")
            axes[2].plot(cum_rewards, label=algorithm)
            axes[2].set_xlabel("Time", fontsize=12)
            axes[2].set_ylabel("Cumulative Rewards of Chosen Arm", fontsize=12)
            axes[2].set_title("Cumulative Rewards of Different Algorithms", y=1.05, fontsize=14)
            axes[2].legend(loc="lower right")
            plt.tight_layout()
    plt.show()