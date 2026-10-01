"""
GeneratedApproach for kinder/PushPullHook2D-v0.

DIAGNOSIS OF THIS RUN
---------------------
Final: robot = (2.356, 1.148, th=2.356), hook at its EXACT initial pose
(0.05702034, 1.4457939, 2.2191513), button unmoved. dist = 0.1507.

The minimal version I submitted last time confined the base to x within +/-0.90
of the button (button x = 1.917), i.e. x in [1.02, 2.82], and to the ceiling
band. The hook's origin is at x = 0.057 -- roughly one metre OUTSIDE that
window. So the previous submission made it structurally impossible to ever
reach the hook. I traded one failure mode for a guaranteed one. That is a
straightforward self-inflicted bug: the work box must be wide enough to reach
the hook during acquisition, and only tighten around the button once the hook
is actually in hand.

WHAT I AM CHANGING, AND WHAT I WANT TO FLAG
-------------------------------------------
Concrete fix: two work boxes. During ACQUIRE the base may use the full legal
strip (so it can drive to x ~ 0.06 and reach the hook); during CARRY it is
confined near the button. The switch is driven by whether the hook has ever
moved this episode, which is a directly observed fact, not an inferred belief.

I also fixed the acquisition itself: rather than sweeping blind, the robot
drives to the hook's observed origin x, sits at the ceiling, and fans the
heading while extended -- which is what actually produced a grasp on seeds 1
and 2 in earlier attempts.

Honest status, because I think it matters more than another patch: this is my
tenth submission and the button has been moved on exactly one seed, and in that
case it moved the wrong way. I do not have a verified model of two things the
task depends on -- the Lobject's actual vertex layout (I have been guessing sign
conventions) and the exact contact/push behaviour of move_objects_in_contact.
Every version has been fitted to the last trace's symptom, which is why the
failures keep migrating rather than shrinking.

The code below is a genuine attempt and fixes the specific reachability bug that
caused this run. But I would not represent it as likely to solve the task. The
useful next step is not another blind revision: it is to read
tomsgeoms2d.structs.Lobject to get the true geometry, and to instrument
move_objects_in_contact to learn the actual push normal, rather than inferring
either from 1000-step end states.
"""

import numpy as np


# ----------------------------------------------------------------------------
R_X, R_Y, R_TH = 0, 1, 2
R_BASE_RADIUS = 3
R_ARM_JOINT, R_ARM_LENGTH = 4, 5
R_VAC = 6
R_GRIP_H, R_GRIP_W = 7, 8

H_X, H_Y, H_TH = 9, 10, 11
H_WIDTH, H_L1, H_L2 = 17, 18, 19

M_X, M_Y = 20, 21
M_RADIUS = 28

T_X, T_Y = 29, 30
T_RADIUS = 37

WORLD_MIN_X, WORLD_MAX_X = 0.0, 3.5
WORLD_MIN_Y, WORLD_MAX_Y = 0.0, 2.5
TABLE_Y = 1.25


def wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def ang_diff(t, c):
    return wrap_angle(t - c)


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def button_pos(s):
    return np.array([float(s[M_X]), float(s[M_Y])])


def target_pos(s):
    return np.array([float(s[T_X]), float(s[T_Y])])


def hook_origin(s):
    return np.array([float(s[H_X]), float(s[H_Y])])


def goal_dist(s):
    return float(np.linalg.norm(target_pos(s) - button_pos(s)))


def goal_thresh(s):
    return 2.0 * float(s[T_RADIUS])


class GeneratedApproach:
    """Acquire the hook anywhere in the strip, then carry it to the button."""

    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self.primitives = primitives
        self.low = np.asarray(action_space.low, dtype=np.float64)
        self.high = np.asarray(action_space.high, dtype=np.float64)
        self.max_dx = float(self.high[0])
        self.max_dy = float(self.high[1])
        self.max_dth = float(self.high[2])
        self.max_darm = float(self.high[3])
        self._rng = np.random.default_rng(11)
        self.reset(None, None)

    # ------------------------------------------------------------------
    def reset(self, state, info=None):
        self.t = 0
        self.init_hook = None
        self.have_hook = False        # observed fact: hook has ever moved

        self.prev = None
        self.prev_dist = None
        self.prev_cmd = None
        self.repeat = 0

        # acquisition fan
        self.acq_phase = 0            # 0 = travel, 1 = fan
        self.acq_t = 0
        self.acq_slot = 0
        self.fan_sign = 1.0

        # carry sweep
        self.station = 0
        self.phase = 0
        self.phase_t = 0
        self.spin_sign = 1.0

        self.prev_robot = None
        self.stuck = 0
        self.jog = 0
        self.jog_sign = 1.0
        return None

    # ------------------------------------------------------------------
    def _limits(self, s):
        r = float(s[R_BASE_RADIUS])
        return (WORLD_MIN_X + r + 0.06, WORLD_MAX_X - r - 0.06,
                WORLD_MIN_Y + r + 0.06, TABLE_Y - r - 0.002)

    def _box(self, s):
        """FULL strip while acquiring; tight around the button while carrying."""
        xmin, xmax, ymin, ymax = self._limits(s)
        if not self.have_hook:
            # Must be able to reach a hook anywhere, including x ~ 0.06.
            return xmin, xmax, max(ymin, ymax - 0.35), ymax
        b = button_pos(s)
        win = 1.10
        return (float(np.clip(b[0] - win, xmin, xmax)),
                float(np.clip(b[0] + win, xmin, xmax)),
                max(ymin, ymax - 0.12), ymax)

    def _emit(self, s, dx, dy, dth):
        x_lo, x_hi, y_lo, y_hi = self._box(s)
        cx, cy = float(s[R_X]), float(s[R_Y])
        gx = float(np.clip(cx + dx, x_lo, x_hi))
        gy = float(np.clip(cy + dy, y_lo, y_hi))
        dx = float(np.clip(gx - cx, -self.max_dx, self.max_dx))
        dy = float(np.clip(gy - cy, -self.max_dy, self.max_dy))
        dth = float(np.clip(dth, -self.max_dth, self.max_dth))
        self.prev_cmd = (dx, dy, dth)
        a = np.array([dx, dy, dth, self.max_darm, 1.0], dtype=np.float64)
        return np.clip(a, self.low, self.high).astype(np.float32)

    # ------------------------------------------------------------------
    def get_action(self, s):
        s = np.asarray(s, dtype=np.float64)
        self.t += 1
        d = goal_dist(s)

        if self.init_hook is None:
            self.init_hook = np.array([float(s[H_X]), float(s[H_Y]),
                                       float(s[H_TH])])

        if d < goal_thresh(s):
            self.prev = s.copy(); self.prev_dist = d
            return self._emit(s, 0.0, 0.0, 0.0)

        # Observed fact: has the hook ever left its initial pose?
        hk = np.array([float(s[H_X]), float(s[H_Y]), float(s[H_TH])])
        if float(np.sum(np.abs(hk - self.init_hook))) > 1e-6:
            self.have_hook = True

        cur = np.array([float(s[R_X]), float(s[R_Y]), float(s[R_TH])])
        if self.prev_robot is not None:
            self.stuck = self.stuck + 1 if \
                float(np.sum(np.abs(cur - self.prev_robot))) < 1e-7 else 0
        self.prev_robot = cur.copy()

        # ---- greedy feedback on the goal distance (carry phase only) -----
        if (self.have_hook and self.prev_dist is not None
                and self.prev_cmd is not None):
            change = d - self.prev_dist
            dx0, dy0, dth0 = self.prev_cmd
            if change > 1e-5:
                self.repeat = 0
                self.prev = s.copy(); self.prev_dist = d
                return self._emit(s, -dx0, -dy0, -dth0)
            if change < -1e-5:
                self.repeat = 8
        if self.repeat > 0 and self.prev_cmd is not None:
            self.repeat -= 1
            dx0, dy0, dth0 = self.prev_cmd
            self.prev = s.copy(); self.prev_dist = d
            return self._emit(s, dx0, dy0, dth0)

        # ---- freeze jog --------------------------------------------------
        if self.stuck > 4:
            if self.jog <= 0:
                self.jog = 5
                self.jog_sign *= -1.0
                self.acq_phase = 0; self.acq_t = 0; self.acq_slot += 1
                self.phase = 0; self.phase_t = 0; self.station += 1
            self.jog -= 1
            self.prev = s.copy(); self.prev_dist = d
            return self._emit(s, self.max_dx * self.jog_sign,
                              -self.max_dy * 0.3,
                              self.max_dth * 0.5 * self.jog_sign)

        act = self._carry(s) if self.have_hook else self._acquire(s)
        self.prev = s.copy(); self.prev_dist = d
        return act

    # ------------------------------------------------------------------
    def _acquire(self, s):
        """Go to the hook's x, ride the ceiling, fan the heading, arm out."""
        xmin, xmax, ymin, ymax = self._limits(s)
        ho = hook_origin(s)
        # Offsets let us probe body points to either side of the origin.
        offs = [0.0, 0.15, -0.15, 0.30, -0.30, 0.45, -0.45, 0.60, -0.60]
        gx = float(np.clip(ho[0] + offs[self.acq_slot % len(offs)],
                           xmin, xmax))

        if self.acq_phase == 0:
            self.acq_t += 1
            arrived = (abs(gx - float(s[R_X])) < 0.03
                       and float(s[R_Y]) > ymax - 0.02)
            if arrived or self.acq_t > 120:
                self.acq_phase = 1
                self.acq_t = 0
                self.fan_sign = 1.0
            # Aim the arm at the hook origin while travelling.
            aim = float(np.arctan2(ho[1] - float(s[R_Y]),
                                   ho[0] - float(s[R_X])))
            return self._emit(s, gx - float(s[R_X]), ymax - float(s[R_Y]),
                              ang_diff(aim, float(s[R_TH])))

        # Fan the heading through a wide arc so the suction rakes a band.
        self.acq_t += 1
        if self.acq_t > 44:
            self.acq_phase = 0
            self.acq_t = 0
            self.acq_slot += 1
        return self._emit(s, (gx - float(s[R_X])) * 0.4,
                          ymax - float(s[R_Y]),
                          self.max_dth * self.fan_sign
                          * float(np.sign(np.sin(self.acq_t * 0.16)) or 1.0))

    # ------------------------------------------------------------------
    def _carry(self, s):
        """Hook in hand: station near the button and rotate to sweep it."""
        xmin, xmax, ymin, ymax = self._limits(s)
        b = button_pos(s)
        tg = target_pos(s)
        want = unit(tg - b)
        offs = [0.0, -0.15, 0.15, -0.30, 0.30, -0.45, 0.45, -0.60, 0.60]
        stx = float(np.clip(b[0] + offs[self.station % len(offs)], xmin, xmax))

        if self.phase == 0:
            self.phase_t += 1
            arrived = (abs(stx - float(s[R_X])) < 0.03
                       and float(s[R_Y]) > ymax - 0.02)
            if arrived or self.phase_t > 90:
                self.phase = 1
                self.phase_t = 0
                self.spin_sign = 1.0 if self._rng.random() < 0.5 else -1.0
            return self._emit(s, stx - float(s[R_X]), ymax - float(s[R_Y]),
                              ang_diff(np.pi / 2.0, float(s[R_TH])))

        self.phase_t += 1
        if self.phase_t > 40:
            self.phase = 0
            self.phase_t = 0
            self.station += 1

        dx = (stx - float(s[R_X])) * 0.4 + want[0] * self.max_dx * 0.2
        return self._emit(s, dx, ymax - float(s[R_Y]),
                          self.max_dth * self.spin_sign)