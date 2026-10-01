"""Reactive controller for BaseMotion3D."""

import numpy as np


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.low = np.asarray(action_space.low, dtype=np.float32)
        self.high = np.asarray(action_space.high, dtype=np.float32)

    def reset(self, state, info):
        self.previous_xy = None
        self.detour = 0
        self.waypoint_x = 0.0

    def get_action(self, state):
        state = np.asarray(state)
        action = np.zeros(11, dtype=np.float32)
        xy = state[0:2]
        target = state[19:21]

        # Most goals are visible across the open floor.  A short partition at
        # y=-1.6 can block the direct path to the few goals behind it.  Detect
        # that case from lack of motion, then go around its nearer end.
        if (self.detour == 0 and self.previous_xy is not None
                and np.linalg.norm(xy - self.previous_xy) < 1e-6
                and np.linalg.norm(target - xy) >= 0.05):
            left, right = -0.85, 1.50
            self.waypoint_x = (left if abs(xy[0] - left) + abs(target[0] - left)
                               < abs(xy[0] - right) + abs(target[0] - right)
                               else right)
            self.detour = 1

        if self.detour == 1:
            action[0] = self.waypoint_x - xy[0]
            if abs(action[0]) < 1e-5:
                self.detour = 2
        if self.detour == 2:
            # Stay just inside the free side of the wall: this remains within
            # the 5 cm goal radius while leaving maximum lateral clearance.
            action[1] = target[1] + 0.045 - xy[1]
            if abs(action[1]) < 1e-5:
                self.detour = 3
        if self.detour == 3:
            # Fine increments are important here: the simulator rejects an
            # entire move whose endpoint overlaps the partition boundary.
            action[0] = np.clip(target[0] - xy[0], -0.02, 0.02)
        elif self.detour == 0:
            action[0:2] = target - xy

        self.previous_xy = xy.copy()
        return np.clip(action, self.low, self.high)
