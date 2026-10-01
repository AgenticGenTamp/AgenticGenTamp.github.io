"""Diagnostic arm basis-motion policy used only by render_policy."""

import numpy as np


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space

    def reset(self, state, info):
        self.t = 0
        r = state.get_object_from_name("robot")
        self.q0 = np.array([state.get(r, "pos_arm_joint%d" % i) for i in range(1, 8)])

    def get_action(self, state):
        a = np.zeros(self.action_space.shape, dtype=self.action_space.dtype)
        r = state.get_object_from_name("robot")
        q = np.array([state.get(r, "pos_arm_joint%d" % i) for i in range(1, 8)])
        # Isolated excursions with return-to-start intervals.
        blocks = [(0, 0.6), (1, 0.6), (2, -0.6), (3, 0.6),
                  (4, 0.6), (5, -0.6), (6, 0.6)]
        block = self.t // 35
        target = self.q0.copy()
        if block < len(blocks) and self.t % 35 >= 10:
            j, delta = blocks[block]
            target[j] += delta
        a[3:10] = np.clip(0.4 * (target - q), -0.1, 0.1)
        a[10] = 1.0
        self.t += 1
        return a
