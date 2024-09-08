#!/usr/bin/env python
# PyTorch

import gymnasium as gym
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import config

sns.set()

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

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
        idx = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        return [self.buffer[i] for i in idx]
    
    def __len__(self):
        return len(self.buffer)

##############################################################################################
# main
env = gym.make('CartPole-v0', render_mode='human')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

main_qn = QNetwork(state_size, action_size)
target_qn = QNetwork(state_size, action_size)
memory = Memory(config.memory_size)

optimizer = optim.Adam(main_qn.parameters(), lr=0.001)
criterion = nn.SmoothL1Loss()

# learning
state, _ = env.reset()
state = np.reshape(state, [1, state_size])
state = torch.FloatTensor(state)

total_step = 0
success_count = 0
steps_history = []
for episode in range(1, config.num_episodes+1):
    step = 0
    target_qn.load_state_dict(main_qn.state_dict())

    # 1 episode
    for _ in range(1, config.max_steps+1):
        step += 1
        total_step += 1

        epsilon = config.e_stop + (config.e_start - config.e_stop) * np.exp(-config.e_decay_rate * total_step)

        # search according to epsilon-greedy
        if epsilon > np.random.rand():
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = np.argmax(main_qn(state).numpy())

        next_state, reward, terminated, truncated, _ = env.step(action)
#        env.render()
        next_state = np.reshape(next_state, [1, state_size])
        next_state = torch.FloatTensor(next_state)

        if terminated or truncated:
            # assignment reward 
            if step >= 190:
                success_count += 1
                reward = 1
            else:
                success_count = 0
                reward = 0

            next_state = torch.zeros(state.shape)

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
            inputs = torch.zeros((config.batch_size, state_size))
            targets = torch.zeros((config.batch_size, action_size))

            minibatch = memory.sample(config.batch_size)

            for i, (state_b, action_b, reward_b, next_state_b) in enumerate(minibatch):
                inputs[i] = state_b
                
                # calculate the value of selected action
                with torch.no_grad():
                    if not torch.all(next_state_b == torch.zeros(state_b.shape)):
                        target = reward_b + config.gamma * torch.max(target_qn(next_state_b))
                    else:
                        target = reward_b

                targets[i] = main_qn(state_b)
                targets[i][action_b] = target

            optimizer.zero_grad()
            output = main_qn(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

        # when finishing episode
        if terminated or truncated:
            break

    print('episode: {}, step: {}, epsilon: {:.4f}'.format(episode, step, epsilon))
    steps_history.append(step)

    if success_count >= 5:
        break

    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    state = torch.FloatTensor(state)

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

