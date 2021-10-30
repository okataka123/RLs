#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

# create a maze
fig = plt.figure(figsize=(3, 3))

# wall
plt.plot([0, 3], [3, 3], color='k')
plt.plot([0, 3], [0, 0], color='k')
plt.plot([0, 0], [0, 2], color='k')
plt.plot([3, 3], [1, 3], color='k')
plt.plot([1, 1], [1, 2], color='k')
plt.plot([2, 3], [2, 2], color='k')
plt.plot([2, 1], [1, 1], color='k')
plt.plot([2, 2], [0, 1], color='k')

# number
for i in range(3):
    for j in range(3):
        plt.text(0.5+i, 2.5-j, str(i+j*3), size=20, ha='center', va='center')
# circle
circle, = plt.plot([0.5], [2.5], marker='o', color='#d3d3d3', markersize=40)
# scale and frames
plt.tick_params(axis='both', which='both', bottom='off', top='off',
        labelbottom='off', right='off', left='off', labelleft='off')
plt.box('off')

# theta-table [up, right, down, left]
theta_0 = np.array([
    [np.nan, 1, 1, np.nan], # 0
    [np.nan, 1, 1, 1], # 1
    [np.nan, np.nan, np.nan, 1], # 2
    [1, np.nan, 1, np.nan], # 3
    [1, 1, np.nan, np.nan], # 4
    [np.nan, np.nan, 1, 1], # 5
    [1, 1, np.nan, np.nan], # 6
    [np.nan, np.nan, np.nan, 1], # 7
])

def get_pi(theta):
    m, n = theta.shape
    pi = np.zeros((m, n))
    exp_theta = np.exp(theta)
    for i in range(m):
        pi[i, :] = exp_theta[i, :]/ np.nansum(exp_theta[i, :])
    pi = np.nan_to_num(pi)
    return pi

pi_0 = get_pi(theta_0)

# get an action according to policy.
def get_a(pi, s):
    return np.random.choice([0, 1, 2, 3], p=pi[s])

# get a stete according to action.
def get_s_next(s, a):
    # up
    if a == 0:
        return s-3
    # right
    elif a == 1:
        return s+1
    # down
    elif a == 2:
        return s+3
    # left
    elif a == 3:
        return s-1

a, b = theta_0.shape
Q = np.random.rand(a, b) * theta_0
print(Q)

def get_a(s, Q, epsilon, pi_0):
    if np.random.rand() < epsilon:
        return np.random.choice([0, 1, 2, 3], p=pi_0[s])
    else:
        return np.nanargmax(Q[s])

def sarsa(s, a, r, s_next, a_next, Q):
    eta = 0.1
    gamma = 0.9
    if s_next == 8:
        Q[s, a] = Q[s, a] + eta * (r - Q[s, a])
    else:
        Q[s, a] = Q[s, a] + eta * (r + gamma * Q[s_next, a_next] - Q[s, a])
    return Q

def q_learning(s, a, r, s_next, a_next, Q):
    eta = 0.1
    gamma = 0.9
    if s_next == 8:
        Q[s, a] = Q[s, a] + eta * (r - Q[s, a])
    else:
        Q[s, a] = Q[s, a] + eta * (r + gamma * np.nanmax(Q[s_next, :]) - Q[s, a])
    return Q

# execute one episode
def play(Q, epsilon, pi, method='sarsa'):
    s = 0
    a = a_next = get_a(s, Q, epsilon, pi)
    s_a_history = [[0, np.nan]]

    while True:
        a = a_next
        s_next = get_s_next(s, a)

        s_a_history[-1][1] = a
        s_a_history.append([s_next, np.nan])

        if s_next == 8:
            r = 1
            a_next = np.nan
        else:
            r = 0
            a_next = get_a(s_next, Q, epsilon, pi)

        if method == 'sarsa':
            Q = sarsa(s, a, r, s_next, a_next, Q)
        else:
            Q = q_learning(s, a, r, s_next, a_next, Q)

        if s_next == 8:
            break
        else:
            s = s_next

    return s_a_history, Q

########################################################
# main
epsilon = 0.5

for episode in range(10):
    epsilon /= 2
    s_a_history, Q = play(Q, epsilon, pi_0, method='sarsa')
    print('episode: {}, step: {}'.format(episode, len(s_a_history)-1))

def animate(i):
    state = s_a_history[i][0]
    circle.set_data((state%3)+0.5, 2.5-int(state/3))
    return circle

anim = animation.FuncAnimation(fig, animate, frames=len(s_a_history), interval=200, repeat=False)
plt.show()
