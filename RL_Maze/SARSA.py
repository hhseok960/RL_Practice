import numpy as np
from maze import Maze

STATE = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']
ACTION = ["up", "right", "down", "left"]

class SARSAAgent():
    """
    수정해야할 사항
    1) epsilon-greedy에서 argmax가 여러 개일 때 이에 대해 랜덤으로 값 반환
    2) learning_rate -> lr로 변수명 변경 (변수명 전체적으로 변경해야한다)
    3) 전체적인 flow 정리
    4) s, s_ 같은 것들 인덱스 뽑는 것들 수정
      - 현재 환경(Maze)이든 에이전트든 state, action에 대한 명확환 분리를 제대로 안해둠
    """
    def __init__(self, env, epsilon=0.9, discount=0.9, learning_rate=0.001):
        self.env = env
        self.state = STATE
        self.action = ACTION
        self.q_func = np.array([[0] * len(self.action)] * (len(self.state) - 1))
        self.policy = self.env.possible_action
        self.q_func *= self.policy  # 미로 정보 중 nan 에 해당하는 것 제거 목적
        self.epsilon = epsilon
        self.discount = discount
        self.learning_rate = learning_rate

    def epsilon_greedy(self, state):
        det = np.random.rand(1)[0]

        if det <= self.epsilon:
            action = self.env.action_space(state)
            action = np.nan_to_num(action)
            action /= action.sum()
            return np.random.choice(self.action, size=1, p=action)[0]
        else:
            current_state = self.state.index(state)
            action = np.nan_to_num(self.q_func[current_state])
            return self.action[np.argmax(action)]

    def SARSA(self, s, a, r_, s_, a_):
        if s_ == "S8":
            self.q_func[s, a] = self.q_func[s, a] + self.learning_rate * (r_ - self.q_func[s, a])

        else:
            self.q_func[s, a] = self.q_func[s, a] + self.learning_rate * (
                r_ + self.discount * self.q_func[s_, a_] - self.q_func[s, a]
            )
