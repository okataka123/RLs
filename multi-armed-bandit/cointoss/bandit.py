#!/usr/bin/env python

import random
import numpy as np

class CoinToss():
    def __init__(self, head_probs, max_episode_steps=30):
        '''
        Args:
            head_probs(list(int)): 各コインの表が出る確率 [p_1, p_2, ...]
        '''
        self.head_probs = head_probs
        self.max_episode_steps = max_episode_steps
        self.toss_count = 0

    def __len__(self):
        return len(self.head_probs)

    def reset(self):
        self.toss_count = 0

    def step(self, action):
        '''
        actionで選択されたコインを投げる。
        action = i で「i番目のコインを投げる」という行動を意味する。
        '''
        final = self.max_episode_steps - 1
        if self.toss_count > final:
            raise Exception('The step count exceed miximun. Prease reset env.')
        else:
            if self.toss_count == final:
                done = True
            else:
                done = False

        if action >= len(self.head_probs):
            raise Exception('The No.{} coin does not exit.'.format(action))
        else:
            head_prob = self.head_probs[action]
            if random.random() < head_prob:
                reward = 1.0
            else:
                reward = 0.0
            self.toss_count += 1
            return reward, done


class EpsilonGreedyAgent():
    def __init__(self, epsilon):
        self.epsilon = epsilon
        # 各コインの期待値
        self.V = [] 

    def policy(self):
        '''
        Epsilon-Greedyでコインを選択し、探索or活用を行う。
        '''
        coins = range(len(self.V))
        if random.random() < self.epsilon:
            # 探索
            return random.choice(coins)
        else:
            # 活用
            return np.argmax(self.V)

    def play(self, env):
        '''
        コイントスゲームをプレイ
        '''
        # N[i] := コインiを投げた回数
        N = [0 for _ in range(len(env))]
        self.V = [0 for _ in range(len(env))]

        env.reset()
        done = False
        rewards = []

        while not done:
            selected_coin = self.policy()
            reward, done = env.step(selected_coin)
            rewards.append(reward)

            n = N[selected_coin]
            coin_average = self.V[selected_coin]
            new_average = (coin_average * n + reward)/(n+1)
            self.V[selected_coin] = new_average
            N[selected_coin] += 1

        return rewards


if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt

    def main():
        # 今回は5枚のコインを使ったゲーム
        env = CoinToss([0.1, 0.5, 0.1, 0.9, 0.1])
        # epsilonをいろいろ変化させた場合をそれぞれ実験する。
        epsilons = [0.0, 0.1, 0.2, 0.5, 0.8] 
        # 1ゲームのmaxのステップ数をいろいろ変えた実験をやる。10〜300回まで。[10, 20, 30, ..., 300]
        game_steps = list(range(10, 310, 10)) 

        result = {}
        for e in epsilons:
            agent = EpsilonGreedyAgent(epsilon=e)
            means = []
            for s in game_steps:
                env.max_episode_steps = s
                reward = agent.play(env)
                means.append(np.mean(reward))
            result['eplislon={}'.format(e)] = means
        result['coin toss count'] = game_steps
        result = pd.DataFrame(result)
        result.set_index('coin toss count', drop=True, inplace=True)
        result.plot.line(figsize=(10, 5))
        plt.show()

    main()

