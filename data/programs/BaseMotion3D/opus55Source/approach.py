import numpy as np

TARGET_RADIUS = 0.05


class GeneratedApproach:
    """Drive the base to the target.

    Base deltas are in the world frame, and each axis is clipped to +-0.4
    separately, so the robot moves along both axes at once. If a move would
    collide, the env rejects the whole step. To handle that, an internal copy
    of the env simulates candidate actions (smaller steps, base rotations)
    and picks the one that gets closest to the target.
    """

    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.max_mag = float(action_space.high[0]) * 0.999
        self._sim = None
        try:
            from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
            self._sim = KinderGeom3DEnv("kinder/BaseMotion3D-v0")
            self._sim.reset(seed=0)
        except Exception:  # pragma: no cover
            self._sim = None

    def reset(self, state, info):
        pass

    def _greedy(self, s, scale=1.0, drot=0.0):
        a = np.zeros(self.action_space.shape, dtype=np.float32)
        dx = s[19] - s[0]
        dy = s[20] - s[1]
        m = self.max_mag * scale
        a[0] = np.clip(dx, -m, m)
        a[1] = np.clip(dy, -m, m)
        a[2] = drot
        return a

    def _sim_step(self, s, a):
        self._sim.set_state(np.asarray(s, dtype=np.float32))
        o, _, term, _, _ = self._sim.step(a)
        return o, term

    def get_action(self, state):
        s = np.asarray(state, dtype=float)
        a0 = self._greedy(s)
        if self._sim is None:
            return a0
        try:
            o, term = self._sim_step(s, a0)
            if term or not np.allclose(o[:2], s[:2]):
                # Full greedy move is collision-free.
                if term or np.hypot(o[19] - o[0], o[20] - o[1]) < np.hypot(
                    s[19] - s[0], s[20] - s[1]
                ):
                    return a0
            best, best_key = a0, None
            m = self.max_mag
            cands = []
            for drot in [0.0, 0.2, -0.2, m, -m, 0.1, -0.1]:
                for scale in [1.0, 0.75, 0.5, 0.3, 0.15, 0.05, 0.0]:
                    cands.append(self._greedy(s, scale, drot))
            # Also try sideways moves (sliding along obstacles).
            for ang in np.linspace(-np.pi, np.pi, 16, endpoint=False):
                for r in [m, m / 2, m / 4]:
                    a = np.zeros(self.action_space.shape, dtype=np.float32)
                    a[0] = np.clip(r * np.cos(ang), -m, m)
                    a[1] = np.clip(r * np.sin(ang), -m, m)
                    cands.append(a)
            d0 = np.hypot(s[19] - s[0], s[20] - s[1])
            for a in cands:
                o, term = self._sim_step(s, a)
                d = np.hypot(o[19] - o[0], o[20] - o[1])
                moved = not np.allclose(o[:3], s[:3])
                key = (0 if term else 1, d if moved else d0 + 1.0)
                if best_key is None or key < best_key:
                    best_key, best = key, a
            return best
        except Exception:
            return a0
