import numpy as np


class GeneratedApproach:
    """Fast feedback script that reaches and opens the island's near drawer."""

    def __init__(self, action_space, observation_space, primitives):
        self.low = np.asarray(action_space.low, dtype=np.float32)
        self.high = np.asarray(action_space.high, dtype=np.float32)
        self.t = 0
        self.home = None
        self.phase = 0
        self.phase_t = 0
        self.hold_pose = None

    def reset(self, state, info):
        self.t = 0
        self.home = np.asarray(state[128:135], dtype=float).copy()
        self.phase = 0
        self.phase_t = 0
        self.hold_pose = None

    def get_action(self, state):
        self.t += 1
        self.phase_t += 1
        s = np.asarray(state)
        a = np.zeros(11, dtype=np.float32)
        a[10] = 1.0

        def base_goal(x, y, yaw):
            error = np.array([x, y, yaw]) - s[125:128]
            a[:3] = np.clip(0.8 * error, -0.1, 0.1)

        drawer_open = float(np.max(s[103:109]))
        if self.hold_pose is not None:
            base_goal(*self.hold_pose)
        elif self.phase == 0:
            base_goal(s[125], -1.35, s[127])
            if self.phase_t >= 20:
                self.phase, self.phase_t = 1, 0
        elif self.phase == 1:
            # The simultaneous diagonal translation and rotation makes the
            # folded forearm catch the c2 front and slide it to its limit.
            base_goal(0.78, -1.35, 1.74)
            if self.phase_t >= 28:
                self.phase, self.phase_t = 2, 0
        elif self.phase == 2:
            base_goal(0.78, -0.92, 1.74)
            if self.phase_t >= 15:
                self.phase, self.phase_t = 3, 0
        elif self.phase == 3:
            # First preload. Strong catches are latched immediately; weak
            # catches are retried from a different collision angle.
            if drawer_open > 0.52:
                self.hold_pose = s[125:128].astype(float).copy()
            elif self.phase_t >= 10 and drawer_open < 0.45:
                self.phase, self.phase_t = 4, 0
            else:
                base_goal(0.94, -0.92, 1.74)
        elif self.phase == 4:
            base_goal(0.80, -1.55, 2.00)
            if self.phase_t >= 15:
                self.phase, self.phase_t = 5, 0
        elif self.phase == 5:
            base_goal(0.80, -0.92, 2.00)
            if self.phase_t >= 17:
                self.phase, self.phase_t = 6, 0
        else:
            if drawer_open > 0.52:
                self.hold_pose = s[125:128].astype(float).copy()
            else:
                base_goal(1.05, -0.92, 2.00)
        return np.clip(a, self.low, self.high)
