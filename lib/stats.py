import numpy as np
import matplotlib.pyplot as plt

def average_rewards(avg_rewards, title="Avg. Rewards",\
                         xlabel="epoch", ylabel="reward"):
    plt.plot(avg_rewards)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig('data/tmp/avg_reward.png', bbox_inches='tight')
    plt.close()
