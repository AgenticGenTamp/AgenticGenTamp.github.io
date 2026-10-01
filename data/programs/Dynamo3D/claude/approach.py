"""Policy for Dynamo3DEnv (variable object count).

Empirical findings (the environment is a black box):

* The reward is a constant -1.0 on every step, everywhere, for every action.
  There is no shaping and no positive reward to be collected, so the episode
  return is simply -(number of steps): the score is maximised by terminating
  as fast as possible.
* ``terminated`` becomes True exactly when the robot base centre
  ``(pos_base_x, pos_base_y)`` enters a fixed axis-aligned goal rectangle on
  the floor.  Chairs are pure obstacles; they happen to spawn on top of the
  goal and are simply shoved aside by the base.
* Two scene layouts were observed:
    - 1 chair   -> goal rectangle x in [0.75, 1.25], y in [-0.25, 0.25]
    - 3 or 12 chairs -> goal rectangle x in [2.25, 2.85], y in [2.25, 2.85]
* Base actions are world-frame position deltas clipped to +-0.1 (about 0.087 m
  of the command is realised per step).  Arm/gripper are irrelevant.

The policy drives the base at maximum speed to the closest point of the
(slightly shrunk) goal rectangle, plowing straight through any chair in the
way, with a side-step recovery if it ever gets wedged.  If the primary goal
guess turns out to be wrong it falls back to the other known rectangle and
then to an expanding lawn-mower search, so it degrades gracefully on unseen
scene layouts.
"""

import re

import numpy as np

MAXD = 0.1


def _idx(name):
    m = re.findall(r'(\d+)', name)
    return int(m[-1]) if m else 0


# known goal rectangles: (xmin, xmax, ymin, ymax)
RECT_A = (0.75, 1.25, -0.25, 0.25)
RECT_B = (2.25, 2.85, 2.25, 2.85)


def _rect_center(r):
    return np.array([0.5 * (r[0] + r[1]), 0.5 * (r[2] + r[3])])


def _nearest_in_rect(r, p, margin=0.09):
    xmin, xmax, ymin, ymax = r
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    xlo, xhi = min(xmin + margin, cx), max(xmax - margin, cx)
    ylo, yhi = min(ymin + margin, cy), max(ymax - margin, cy)
    return np.array([float(np.clip(p[0], xlo, xhi)),
                     float(np.clip(p[1], ylo, yhi))])


class GeneratedApproach:

    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self._robot = None
        self.t = 0
        try:
            self.adim = int(action_space.shape[0])
        except Exception:
            self.adim = 11

    # ------------------------------------------------------------- helpers
    def _robot_obj(self, state):
        if self._robot is not None:
            return self._robot
        obj = None
        try:
            obj = state.get_object_from_name('robot')
            state.get(obj, 'pos_base_x')
        except Exception:
            obj = None
            for name in state.get_object_names():
                try:
                    o = state.get_object_from_name(name)
                    state.get(o, 'pos_base_x')
                    obj = o
                    break
                except Exception:
                    continue
        self._robot = obj
        return obj

    def _robot_xy(self, state):
        r = self._robot_obj(state)
        return np.array([float(state.get(r, 'pos_base_x')),
                         float(state.get(r, 'pos_base_y'))])

    def _chairs(self, state):
        out = []
        rob = self._robot_obj(state)
        for name in state.get_object_names():
            o = state.get_object_from_name(name)
            if rob is not None and o == rob:
                continue
            try:
                xy = np.array([float(state.get(o, 'x')), float(state.get(o, 'y'))])
            except Exception:
                continue
            if not np.all(np.isfinite(xy)) or np.any(np.abs(xy) > 50.0):
                continue
            out.append((_idx(name), xy))
        out.sort(key=lambda t: t[0])
        return [xy for _, xy in out]

    # ---------------------------------------------------------------- plan
    def reset(self, state, info):
        self.t = 0
        self._robot = None
        try:
            b0 = self._robot_xy(state)
        except Exception:
            b0 = np.zeros(2)
        chairs = self._chairs(state)
        n = len(chairs)

        # decide which known rectangle is the goal.  Prefer the layout whose
        # centre has a chair sitting close to it; fall back on the chair count.
        def near(rect):
            c = _rect_center(rect)
            if not chairs:
                return 1e9
            return min(float(np.linalg.norm(c - p)) for p in chairs)

        da, db = near(RECT_A), near(RECT_B)
        if da < 0.9 and db >= 0.9:
            order = [RECT_A, RECT_B]
        elif db < 0.9 and da >= 0.9:
            order = [RECT_B, RECT_A]
        elif n == 1:
            order = [RECT_A, RECT_B]
        else:
            order = [RECT_B, RECT_A]
        self.rects = order
        self.phase = 0          # index into self.rects, then fallback search
        self.phase_start = 0
        self.fallback_wps = None
        self.fwi = 0
        self.prev = b0.copy()
        self.stuck = 0
        self.detour = None
        self.detour_left = 0
        self.chairs0 = chairs
        self.b0 = b0

    # -------------------------------------------------------------- search
    def _build_fallback(self, b):
        """Expanding lawn-mower over the plausible workspace."""
        wps = []
        centers = [_rect_center(r) for r in self.rects] + list(self.chairs0)
        seen = []
        for c in centers:
            if any(float(np.linalg.norm(c - s)) < 0.35 for s in seen):
                continue
            seen.append(c)
            for rad in (0.0, 0.35, 0.7):
                if rad == 0.0:
                    wps.append(np.array(c, dtype=float))
                    continue
                k = 6 if rad < 0.5 else 10
                for i in range(k):
                    a = 2.0 * np.pi * i / k
                    wps.append(np.array(c, dtype=float)
                               + rad * np.array([np.cos(a), np.sin(a)]))
        # coarse lawn-mower over the whole plausible arena
        y = -1.0
        up = True
        while y <= 3.6:
            xs = np.arange(-1.0, 3.61, 0.35)
            if not up:
                xs = xs[::-1]
            for x in xs:
                wps.append(np.array([x, y]))
            up = not up
            y += 0.35
        return wps

    # -------------------------------------------------------------- action
    def get_action(self, state):
        self.t += 1
        a = np.zeros(self.adim, dtype=np.float32)
        try:
            b = self._robot_xy(state)
        except Exception:
            return a

        moved = float(np.linalg.norm(b - self.prev))
        self.prev = b.copy()
        if moved < 0.012:
            self.stuck += 1
        else:
            self.stuck = 0

        # ---- pick the current target -----------------------------------
        target = None
        if self.phase < len(self.rects):
            rect = self.rects[self.phase]
            target = _nearest_in_rect(rect, b)
            arrived = float(np.linalg.norm(target - b)) < 0.05
            # if we sat inside the guessed rectangle for a while and nothing
            # happened, the guess was wrong -> move on
            if (arrived and self.t - self.phase_start > 25) or \
               (self.t - self.phase_start > 320):
                self.phase += 1
                self.phase_start = self.t
                self.stuck = 0
                if self.phase >= len(self.rects):
                    self.fallback_wps = self._build_fallback(b)
                    self.fwi = 0
                return a
        else:
            if self.fallback_wps is None:
                self.fallback_wps = self._build_fallback(b)
                self.fwi = 0
            while True:
                if self.fwi >= len(self.fallback_wps):
                    self.fwi = 0
                target = self.fallback_wps[self.fwi]
                if float(np.linalg.norm(target - b)) < 0.12 or self.stuck > 25:
                    self.fwi += 1
                    self.stuck = 0
                    continue
                break

        # ---- wedged?  try a short sideways detour ----------------------
        if self.detour_left > 0:
            step = self.detour
            self.detour_left -= 1
        else:
            d = target - b
            nrm = float(np.linalg.norm(d))
            # the action box is per-axis: scale by the L-inf norm so that
            # diagonal motion uses the full +-0.1 on BOTH axes (41% faster)
            linf = float(np.max(np.abs(d)))
            if linf < 1e-9:
                step = np.zeros(2)
            else:
                step = d * (MAXD / linf) if linf > MAXD else d
            if self.stuck >= 12:
                u = d / nrm if nrm > 1e-6 else np.array([1.0, 0.0])
                p = np.array([-u[1], u[0]])
                if (self.t // 37) % 2:
                    p = -p
                self.detour = (p * 0.9 - u * 0.35) * MAXD
                self.detour_left = 14
                step = self.detour
                self.stuck = 0

        a[0] = float(np.clip(step[0], -MAXD, MAXD))
        a[1] = float(np.clip(step[1], -MAXD, MAXD))
        return a
