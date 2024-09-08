import gymnasium as gym
from gymnasium import spaces
import numpy as np

#class CustomEnv1(gym.Env):
#    def __init__(self):
#        super(CustomEnv1, self).__init__()
#        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
#        self.action_space = spaces.Discrete(2)
#        self.state = None
#
#    def reset(self):
#        self.state = np.random.rand(4)
#        return self.state
#
#    def step(self, action):
#        # ここに環境の動作ロジックを記述します
#        reward = 1.0  # 仮の報酬
#        done = False  # 終了条件
#        self.state = np.random.rand(4)  # 新しい状態
#        return self.state, reward, done, {}
#
#    def render(self, mode='human'):
#        print(f"State: {self.state}")
#

NUM_PATTERN = 3 # 受信パターン

class SimpleBattleship(gym.Env):
    '''
    まずはpatternを考慮せずに考えてみる。
    '''
    def __init__(self, grid_size=10):
        super().__init__()
        self.grid_size = grid_size
#        self.action_space = spaces.Discrete(grid_size * grid_size * NUM_PATTERN)
        self.action_space = spaces.MultiDiscrete([grid_size, grid_size])
        self.observation_space = spaces.MultiDiscrete([grid_size, grid_size])


#        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        self.state = None

        self.reset()

    def reset(self):
        self.state = np.random.rand(4)
        return self.state

    def step(self, action):
        # ここに環境の動作ロジックを記述します
        reward = 1.0  # 仮の報酬
        done = False  # 終了条件
        self.state = np.random.rand(4)  # 新しい状態
        return self.state, reward, done, {}

    def render(self, mode='human'):
        print(f"State: {self.state}")


if __name__ == '__main__':
    env = CustomEnv1()

