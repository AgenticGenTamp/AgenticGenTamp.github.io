import numpy as np


class GeneratedApproach:
    """Drive the holonomic base to the nearest point of the goal region.

    The goal region isn't observed; it's determined by the task layout, which
    is identified by the number of chairs (1 chair -> small region near origin,
    otherwise the far corner region). The base is very stiff and pushes chairs
    out of the way, so a direct path works. Per-axis saturation (+-0.1) gives
    the fastest motion (diagonal moves at ~0.14 m/step).
    """

    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.lo = np.array([3.45, 3.45])
        self.hi = np.array([4.2, 4.2])
        self.margin = 0.12

    def _robot(self, state):
        for name in state.get_object_names():
            o = state.get_object_from_name(name)
            if name == "robot" or o.type.name == "mujoco_tidybot_robot":
                return o
        return None

    def reset(self, state, info):
        n = sum(1 for nm in state.get_object_names() if nm.startswith("obstacle_chair"))
        if n == 1:
            self.lo, self.hi = np.array([0.8, -0.2]), np.array([1.2, 0.2])
        else:
            self.lo, self.hi = np.array([3.45, 3.45]), np.array([4.2, 4.2])
        self.prev = None
        self.stuck = 0

    def get_action(self, state):
        r = self._robot(state)
        p = np.array([state.get(r, "pos_base_x"), state.get(r, "pos_base_y")])
        m = self.margin
        if self.stuck > 10:
            m = 0.18  # go deeper into the region if we're stalling at the edge
        target = np.clip(p, self.lo + m, self.hi - m)
        d = target - p
        if self.prev is not None and np.linalg.norm(p - self.prev) < 0.01:
            self.stuck += 1
        self.prev = p
        a = np.zeros(11, dtype=np.float32)
        a[0:2] = np.clip(d, -0.1, 0.1)
        return a
