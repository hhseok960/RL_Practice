import numpy as np
from maze import Maze

STATE = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']
ACTION = ["up", "right", "down", "left"]

class PolicyGradientAgent():
    def __init__(self, env):
        self.env = env
        self.action = ACTION
        self.state = STATE
        self.policy = self.env.possible_action

    def get_softmax_policy(self, state):
        assert state != "S8", "S8 is terminal state!"
        softmax_policy = self.env.action_space(state)
        softmax_policy = np.exp(softmax_policy)
        softmax_policy = np.nan_to_num(softmax_policy)
        softmax_policy /= softmax_policy.sum()
        return softmax_policy

    def get_action(self, state):
        policy = self.get_softmax_policy(state)
        return np.random.choice(self.action, size=1, p=policy)[0]

    def exploration(self):
        current_state = self.env.reset()
        path = []
        reward = []
        while True:
            action = self.get_action(current_state)
            path.append([current_state, action])
            next_state = self.env.return_next_state(current_state, action)
            reward.append(self.env.get_reward(current_state, action))
            current_state = next_state
            if next_state == "S8":
                path.append([next_state, np.nan])
                print("Arrive at terminate point!")
                break
        return path, reward

    def policy_update(self):
        path, reward = self.exploration()
        learning_rate = 0.1
        num_step = len(path) - 1
        delta_theta = self.policy.copy()

        for i in range(delta_theta.shape[0]):
            pi = self.get_softmax_policy(state=self.state[i])
            for j in range(delta_theta.shape[1]):
                if not(np.isnan(self.policy[i, j])):
                    s_i_a = [SA for SA in path if SA[0] == self.state[i]]
                    s_i_a_j = [SA for SA in path if SA == [self.state[i], self.action[j]]]
                    N_i = len(s_i_a)
                    N_ij = len(s_i_a_j)
                    delta_theta[i, j] = (N_ij - pi[j] * N_i) / num_step
        new_policy = self.policy + (learning_rate * delta_theta)
        self.policy = new_policy
        return self.policy

