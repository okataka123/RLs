#!/usr/bin/env python
# tensorflow 1.X

import gym
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
from tensorflow.losses import huber_loss
import config
sns.set()

class QNetwork:
    def __init__(self, state_size, action_size):
        self.model = Sequential()
        self.model.add(Dense(16, activation='relu', input_dim=state_size))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(action_size, activation='linear'))
        self.model.compile(loss=huber_loss, optimizer=Adam(lr=0.001))

class Memory:
    """
    replay buffer
    """
    def __init__(self, memory_size):
        self.buffer = deque(maxlen=memory_size)

    def add(self, experience):
        """
        Adding experience 
        """
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[i] for i in idx]
    
    def __len__(self):
        return len(self.buffer)

##############################################################################################
# main
env = gym.make('CartPole-v0')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

main_qn = QNetwork(state_size, action_size)
target_qn = QNetwork(state_size, action_size)
memory = Memory(config.memory_size)

# learning
state = env.reset()
state = np.reshape(state, [1, state_size])
#print('state =', state)

total_step = 0
success_count = 0
steps_history = []
for episode in range(1, config.num_episodes+1):
    step = 0
    target_qn.model.set_weights(main_qn.model.get_weights())

    # 1 eposode
    for _ in range(1, config.max_steps+1):
        step += 1
        total_step += 1

        epsilon = config.e_stop + (config.e_start - config.e_stop) * np.exp(-config.e_decay_rate * total_step)

        # search according to epsilon-greedy
        if epsilon > np.random.rand():
            action = env.action_space.sample()
        else:
            action = np.argmax(main_qn.model.predict(state)[0])

        next_state, _, done, _ = env.step(action)
        env.render()
        next_state = np.reshape(next_state, [1, state_size])

        if done:
            # assignment reward 
            if step >= 190:
                success_count += 1
                reward = 1
            else:
                success_count = 0
                reward = 0

            next_state = np.zeros(state.shape)

            # add this experience to replay buffer
            if step > config.warmup:
                memory.add((state, action, reward, next_state))

        else:
            reward = 0
            if step > config.warmup:
                memory.add((state, action, reward, next_state))

            state = next_state

        # update q_value
        if len(memory) >= config.batch_size:
            inputs = np.zeros((config.batch_size, 4))
            targets = np.zeros((config.batch_size, 2))

            minibatch = memory.sample(config.batch_size)

            for i, (state_b, action_b, reward_b, next_state_b) in enumerate(minibatch):
                inputs[i] = state_b
                
                # calculate the value of selected action
                if not (next_state_b == np.zeros(state_b.shape)).all(axis=1):
                    target = reward_b + config.gamma * np.amax(target_qn.model.predict(next_state_b)[0])
                else:
                    target = reward_b
            
                targets[i] = main_qn.model.predict(state_b)
                targets[i][action_b] = target

            main_qn.model.fit(inputs, targets, epochs=1, verbose=0)

        # when finishing spisode
        if done:
            break

    print('episode: {}, step: {}, epsilon: {:.4f}'.format(episode, step, epsilon))
    steps_history.append(step)

    if success_count >= 5:
        break

    state = env.reset()
    state = np.reshape(state, [1, state_size])

def visualize():
    """
    visualization of learning convergence
    """
    fig, ax = plt.subplots()
    ax.plot(list(range(1, len(steps_history)+1)), steps_history)
    ax.set_xlabel('episode')
    ax.set_ylabel('steps')
    fig.savefig('convergence.png')

visualize()

