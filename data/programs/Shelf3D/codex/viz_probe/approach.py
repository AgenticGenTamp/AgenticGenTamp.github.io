import numpy as np


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.n = 0

    def reset(self, state, info):
        self.n = 0

    def get_action(self, state):
        self.n += 1
        a = np.zeros(11, np.float32)
        robot = state.get_object_from_name("robot")
        target = (-.12, 2.212, -2.838, 0.0, -.304, -.857, -.267)
        # Establish the alternate q3/q5 branch before unfolding q2/q4.
        active = (0, 2, 4, 5, 6) if self.n < 120 else range(7)
        for j in active:
            q = float(state.get(robot, "pos_arm_joint%d" % (j + 1)))
            a[3 + j] = np.clip(2 * (target[j] - q), -.1, .1)
        return a
