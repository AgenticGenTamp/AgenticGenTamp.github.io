"""Visual diagnostic: toggle only the gripper."""
import numpy as np

class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.space = action_space
    def reset(self, state, info):
        self.t = 0
    def get_action(self, state):
        a = np.zeros(self.space.shape, dtype=self.space.dtype)
        a[10] = 0.0 if self.t < 15 else 1.0
        self.t += 1
        return a
