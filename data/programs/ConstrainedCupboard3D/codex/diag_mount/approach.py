import numpy as np


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.a = action_space
        self.poses = np.array([
            [-.916, -1.379, 1.394, -1.651, -.586, -.843, 1.570],
            [-.918, -1.2194, 1.590, -1.5362, -.5576, -.7782, 1.570],
            [-.8544, -1.1845, 1.6079, -1.3973, -.5757, -.6972, 1.570],
            [-.8655, -1.0794, 1.7041, -1.3278, -.5649, -.657, 1.570],
        ])

    def reset(self, state, info):
        self.step = 0

    def get_action(self, state):
        a = np.zeros(self.a.shape, dtype=self.a.dtype)
        r = state.get_object_from_name("robot")
        q = np.array([float(state.get(r, f"pos_arm_joint{i}")) for i in range(1, 8)])
        target = self.poses[min(self.step // 60, len(self.poses)-1)]
        self.step += 1
        e = target - q
        e[[0, 2, 4, 6]] = (e[[0, 2, 4, 6]] + np.pi) % (2*np.pi) - np.pi
        a[3:10] = np.clip(.7*e, -.1, .1)
        a[10] = 1.
        return a
