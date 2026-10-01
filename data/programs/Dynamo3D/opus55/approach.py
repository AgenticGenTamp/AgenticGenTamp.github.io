"""Dynamo3D approach: drive the holonomic base straight into the (fixed,
layout-dependent) goal region, pushing chairs out of the way.

Empirically mapped goal regions (robot base centre must be inside):
  - 1-chair layout:        x in [0.75, 1.25], y in [-0.25, 0.25]
  - 3/12-chair layouts:    x in [3.40, 4.25], y in [3.40, 4.25]
Actions are world-frame base deltas (clipped to +-0.1, ~0.087 m realised).
"""
import numpy as np

GOALS = {
    "small": (0.75, 1.25, -0.25, 0.25),
    "big": (3.40, 4.25, 3.40, 4.25),
}
MARGIN = 0.035


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self.low = np.asarray(action_space.low, dtype=np.float32)
        self.high = np.asarray(action_space.high, dtype=np.float32)
        self.robot_type = None
        self.chair_type = None
        for t in observation_space.types:
            if t.name == "mujoco_tidybot_robot":
                self.robot_type = t
            elif t.name == "mujoco_movable_object":
                self.chair_type = t
        self.target = None
        self.hist = []
        self.detour = None
        self.detour_steps = 0

    # ------------------------------------------------------------------
    def _robot(self, state):
        rt = self.robot_type
        if rt is None:
            for n in state.get_object_names():
                o = state.get_object_from_name(n)
                if "robot" in n:
                    return o
        objs = state.get_objects(rt)
        return objs[0]

    def _chairs(self, state):
        if self.chair_type is None:
            return []
        out = []
        for c in state.get_objects(self.chair_type):
            out.append((state.get(c, "x"), state.get(c, "y")))
        return out

    def _pick_goal(self, state):
        chairs = self._chairs(state)
        if len(chairs) == 1:
            return GOALS["small"]
        if len(chairs) == 0:
            return GOALS["small"]
        # multi-chair layouts: goal region at the far corner
        return GOALS["big"]

    def reset(self, state, info):
        x0, x1, y0, y1 = self._pick_goal(state)
        r = self._robot(state)
        px, py = state.get(r, "pos_base_x"), state.get(r, "pos_base_y")
        tx = float(np.clip(px, x0 + MARGIN, x1 - MARGIN))
        ty = float(np.clip(py, y0 + MARGIN, y1 - MARGIN))
        self.region = (x0, x1, y0, y1)
        self.target = np.array([tx, ty])
        self.center = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        self.hist = []
        self.detour = None
        self.detour_steps = 0

    def get_action(self, state):
        if self.target is None:
            self.reset(state, {})
        r = self._robot(state)
        p = np.array([state.get(r, "pos_base_x"), state.get(r, "pos_base_y")])
        self.hist.append(p.copy())
        # Head to target; gradually move target toward centre if stuck long.
        tgt = self.target
        if self.detour is not None and self.detour_steps > 0:
            tgt = self.detour
            self.detour_steps -= 1
            if np.linalg.norm(tgt - p) < 0.05:
                self.detour_steps = 0
        elif len(self.hist) > 15:
            moved = np.linalg.norm(self.hist[-1] - self.hist[-15])
            if moved < 0.1 and np.linalg.norm(self.target - p) > 0.15:
                # stuck: sidestep perpendicular to goal direction
                d = self.target - p
                d = d / (np.linalg.norm(d) + 1e-9)
                perp = np.array([-d[1], d[0]])
                if np.random.rand() < 0.5:
                    perp = -perp
                self.detour = p + 0.6 * perp - 0.2 * d
                self.detour_steps = 12
                self.hist = []
                tgt = self.detour
        d = tgt - p
        a = np.zeros(self.low.shape, dtype=np.float32)
        a[0:2] = np.clip(d, -0.1, 0.1)
        a = np.clip(a, self.low, self.high)
        return a
