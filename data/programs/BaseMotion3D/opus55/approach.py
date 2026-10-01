import numpy as np

R = 0.044  # termination radius ~0.05 (with margin)
STEP = 0.4
ROTS = [0.8, -0.8, 0.4, -0.4, 2.0, -2.0, 2.4, -2.4, 1.2, -1.2, 1.57, -1.57, 3.14, 0.0]


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def plan_goal(pos, tgt, r=R):
    best, bestc = tgt, np.inf
    for th in np.linspace(0, 2 * np.pi, 360, endpoint=False):
        for rr in (r, 0.0):
            g = tgt + rr * np.array([np.cos(th), np.sin(th)])
            d = np.abs(g - pos)
            n = np.ceil(np.max(d) / STEP - 1e-9)
            c = n * 10 + np.max(d)
            if c < bestc:
                bestc, best = c, g
    return best


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.tgt = None

    def reset(self, state, info):
        state = np.asarray(state, dtype=float)
        self.tgt = state[19:21].copy()
        self.state = state
        self.gen = self._script()

    # ---- helpers (generators yielding base deltas [dx,dy,drot]) ----
    def _move_to(self, goal, min_scale=0.01, rot=None):
        """Move toward goal; on block halve the step. Returns True if reached."""
        scale = 1.0
        while True:
            pos = self.state[:2]
            d = goal - pos
            if np.linalg.norm(d) < 1e-4:
                return True
            step = np.clip(d, -STEP * scale, STEP * scale)
            dr = 0.0
            if rot is not None and scale >= 1.0:
                dr = float(np.clip(wrap(rot - self.state[2]), -STEP, STEP))
            before = self.state[:3].copy()
            yield np.array([step[0], step[1], dr])
            if np.linalg.norm(self.state[:3] - before) < 1e-7:
                scale *= 0.5
                if scale < min_scale:
                    return False
            # keep scale after success (no growth needed for rare case)

    def _rotate_to(self, r):
        scale = 1.0
        while True:
            dr = wrap(r - self.state[2])
            if abs(dr) < 1e-3:
                return True
            before = self.state[:3].copy()
            yield np.array([0.0, 0.0, np.clip(dr, -STEP * scale, STEP * scale)])
            if np.linalg.norm(self.state[:3] - before) < 1e-7:
                scale *= 0.5
                if scale < 0.02:
                    return False

    def _script(self):
        tgt = self.tgt
        goal = plan_goal(self.state[:2], tgt)
        # near the corners of the known obstacle at y~-2.2, x in [-0.7,1.4],
        # a 45-degree base rotation lets the square base get closer
        rot = None
        if tgt[1] < -1.85 and (tgt[0] > 1.2 or tgt[0] < -0.55) and tgt[0] < 1.7 and tgt[0] > -1.0:
            rot = 0.785
        ok = yield from self._move_to(goal, rot=rot)
        # direct approach failed/blocked: try to get closer, then rotations
        yield from self._move_to(tgt.copy())
        for r in ROTS * 3:
            pos = self.state[:2].copy()
            away = pos - tgt
            n = np.linalg.norm(away)
            away = away / n if n > 1e-6 else np.array([1.0, 0.0])
            for back in (0.3, 0.6):
                yield from self._move_to(pos + away * back)
                rok = yield from self._rotate_to(r)
                if rok:
                    break
            yield from self._move_to(tgt.copy(), min_scale=0.005)
            # try axis slides
            yield from self._move_to(np.array([tgt[0], self.state[1]]), 0.005)
            yield from self._move_to(np.array([self.state[0], tgt[1]]), 0.005)
        while True:
            yield np.zeros(3)

    def get_action(self, state):
        state = np.asarray(state, dtype=float)
        if self.tgt is None or np.linalg.norm(state[19:21] - self.tgt) > 1e-6:
            self.reset(state, {})
        self.state = state
        try:
            b = next(self.gen)
        except StopIteration:
            b = np.zeros(3)
        a = np.zeros(11, dtype=np.float32)
        a[:3] = b
        return a
