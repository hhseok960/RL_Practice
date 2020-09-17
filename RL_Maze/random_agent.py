import numpy as np
from maze import Maze

ACTION = ["up", "right", "down", "left"]

class RandomAgent():
    def __init__(self, env):
        # 환경에 대한 객체 선언
        self.env = env
        self.action = ACTION

    def get_policy(self, state):
        assert state != "S8", "S8 is terminal state!"
        random_policy = self.env.action_space(state)
        random_policy = np.nan_to_num(random_policy)
        random_policy /= random_policy.sum()
        return random_policy

    def get_action(self, state):
        policy = self.get_policy(state)
        return np.random.choice(self.action, size=1, p=policy)[0]

    def exploration(self):
        current_state = self.env.reset()
        path = [current_state]
        reward = []
        while True:
            action = self.get_action(current_state)
            next_state = self.env.return_next_state(current_state, action)
            path.append(next_state)
            reward.append(self.env.get_reward(current_state, action))
            current_state = next_state
            if current_state == "S8":
                print("Arrive at terminate point")
                break
        return path, reward

