import numpy as np
import matplotlib.pyplot as plt

# 미로 상태(순서대로 [up , right, down, left])
maze = np.array([[np.nan, 1, 1, np.nan],  # s0
                [np.nan, 1, np.nan, 1],  # s1
                [np.nan, np.nan, 1, 1],  # s2
                [1, 1, 1, np.nan],  # s3
                [np.nan, np.nan, 1, 1],  # s4
                [1, np.nan, np.nan, np.nan],  # s5
                [1, np.nan, np.nan, np.nan],  # s6
                [1, 1, np.nan, np.nan],  # s7、※s8은 목표지점이므로 정책이 없다
                ])
state_name = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']
TRANSITION_PROB = 1


class Maze:
    """
    인자로 받는 state는 모두 str 타입 -> 메서드에서는 따라서 인덱스로 변환해서 사용
    (추후에 variable name 전체적으로 수정)
    """
    def __init__(self):
        self.state = state_name
        self.reward = [-1] * len(self.state)  # 모든 행동에 따라 -1의 보상(벌점)
        self.reward[-1] = 10  # 최종 상태에 도달하면 10의 보상 제공

        self.possible_action = maze
        self.transition_probability = TRANSITION_PROB

    def plot_maze(self, state):
        fig = plt.figure(figsize=(5, 5))
        ax = plt.gca()

        # 붉은 벽 위치
        plt.plot([1, 1], [0, 1], color='red', linewidth=2)
        plt.plot([1, 2], [2, 2], color='red', linewidth=2)
        plt.plot([2, 2], [2, 1], color='red', linewidth=2)
        plt.plot([2, 3], [1, 1], color='red', linewidth=2)

        # 상태를 의미하는 문자열(S0~S8) 표시
        loc = [0.5, 1.5, 2.5]
        for i in range(3):
            for j in range(3):
                plt.text(loc[j], loc[2 - i], self.state[j + 3 * i], size=14, ha='center')

        plt.text(0.5, 2.3, 'START', ha='center')
        plt.text(2.5, 0.3, 'GOAL', ha='center')

        # 그림을 그릴 범위 및 눈금 제거 설정
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 3)
        plt.tick_params(axis='both', which='both', bottom=False, top=False,
                        labelbottom=False, right=False, left=False, labelleft=False)

        # S0에 녹색 원으로 현재 위치를 표시
        state_x = (state % 3) + 0.5
        state_y = 2.5 - (state // 3)
        line, = ax.plot([state_x], [state_y], marker="*", color='g', markersize=60)

    def reset(self):
        return self.state[0]

    def action_space(self, state):
        state = self.state.index(state)
        return self.possible_action[state]

    def get_reward(self, state, action):
        next_state = self.return_next_state(state, action)
        next_state_idx = self.state.index(next_state)
        print("Next State: {0}(index : {1})".format(next_state, next_state_idx))
        return self.reward[next_state_idx]

    def return_next_state(self, state, action):
        state = self.state.index(state)
        if action == "up":
            next_state = state - 3
        elif action == "right":
            next_state = state + 1
        elif action == "down":
            next_state = state + 3
        elif action == "left":
            next_state = state - 1
        return self.state[next_state]





