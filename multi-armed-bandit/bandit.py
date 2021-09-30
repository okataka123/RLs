#!/usr/bin/env python
import numpy as np 
import random
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

class SlotArm:
    def __init__(self, p):
        self.p = p

    def draw(self):
        if self.p > random.random():
            return 1.0
        else:
            return 0.0

class EpsilonGreedy:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def initialize(self, n_arms):
        self.n = np.zeros(n_arms) # Number of trials of arms
        self.v = np.zeros(n_arms) # values of arms

    def select_arm(self):
        """
        choose an arm according to the policy.
        """
        if self.epsilon > random.random():
            return np.random.randint(0, len(self.v))
        else:
            return np.argmax(self.v)

    def update(self, chosen_arm, reward, t):
        """
        update parameters.
        """
        self.n[chosen_arm] += 1
        n = self.n[chosen_arm]
        v = self.v[chosen_arm]
        self.v[chosen_arm] = ((n-1)/float(n))*v + (1/float(n))*reward

    def label(self):
        return 'epsilon-greedy(' + str(self.epsilon) + ')'

class UCB1:
    def initialize(self, n_arms):
        self.n = np.zeros(n_arms) # Number of trials of arms
        self.w = np.zeros(n_arms) # Number of success count 
        self.v = np.zeros(n_arms) # values of armsA

    def select_arm(self):
        """
        choose an arm according to the policy.
        """
        for i in range(len(self.n)):
            if self.n[i] == 0:
                return i
        return np.argmax(self.v)
    
    def update(self, chosen_arm, reward, t):
        self.n[chosen_arm] += 1
        if reward == 1.0:
            self.w[chosen_arm] += 1

        for i in range(len(self.n)):
            if self.n[i] == 0:
                return 

        for i in range(len(self.v)):
            self.v[i] = self.w[i]/self.n[i] + (2*math.log(t)/self.n[i])**0.5

    def label(self):
        return 'ucb1'

# execute simulation
def play(algo, arms, num_sims, num_time):
    # history
    times = np.zeros(num_sims * num_time) # number of games
    rewards = np.zeros(num_sims * num_time)

    for sim in range(num_sims):
        algo.initialize(len(arms))

        for time in range(num_time):
            index = sim * num_time + time

            times[index] = time + 1
            chosen_arm = algo.select_arm()
            reward = arms[chosen_arm].draw()
            rewards[index] = reward

            algo.update(chosen_arm, reward, time+1)

    return [times, rewards]

def main():
    arms = (SlotArm(0.3), SlotArm(0.5), SlotArm(0.9))
    algos = (EpsilonGreedy(0.1), UCB1())

    for algo in algos:
        results = play(algo, arms, 1000, 250)

        df = pd.DataFrame({'times': results[0], 'rewards': results[1]})
        mean = df['rewards'].groupby(df['times']).mean()
        plt.plot(mean, label=algo.label())

    plt.xlabel('Step')
    plt.ylabel('Average Reward')
    plt.legend(loc='best')
    plt.show()

main()
