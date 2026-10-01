"""Sweep policy: push floor cubes with the arm tip toward the -y goal band."""
import numpy as np

BASE_GAIN = 0.87
ROT_GAIN = 0.994
ARM_GAIN = 0.25

# Empirically calibrated pushing configuration (arm tip pressed to the floor,
# straight ahead of the base) and the resulting contact point in the base frame.
Q_PUSH = np.array([0.0, 1.8838, 3.1416, -1.5828, 0.0, 0.3250, 1.5708])
CONTACT_FWD = 0.32
CONTACT_LAT = -0.04

Y_GOAL = 0.05
PUSH_SPEED = 0.026
BIASES = [0.045, -0.07, 0.09, -0.11]


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self._cube_type = None
        self._robot_type = None
        for t in observation_space.types:
            if t.name == "mujoco_movable_object":
                self._cube_type = t
            elif t.name == "mujoco_tidybot_robot":
                self._robot_type = t

    # ---------------- state helpers ----------------
    def _robot(self, state):
        r = state.get_object_from_name("robot")
        j = np.array([float(state.get(r, "pos_arm_joint%d" % i)) for i in range(1, 8)])
        b = np.array([float(state.get(r, f)) for f in
                      ("pos_base_x", "pos_base_y", "pos_base_rot")])
        return j, b

    def _cubes(self, state):
        out = {}
        if self._cube_type is None:
            for name in sorted(state.get_object_names()):
                if name.startswith("cube"):
                    o = state.get_object_from_name(name)
                    out[name] = np.array([float(state.get(o, "x")),
                                          float(state.get(o, "y"))])
            return out
        for o in state.get_objects(self._cube_type):
            if not o.name.startswith("cube"):
                continue
            out[o.name] = np.array([float(state.get(o, "x")), float(state.get(o, "y"))])
        return out

    # ---------------- planning ----------------
    def reset(self, state, info):
        self.t = 0
        cubes = self._cubes(state)
        self.order = sorted(cubes.keys(), key=lambda n: cubes[n][1])
        self.idx = 0
        self.bias_i = 0
        self.phase = "goto"
        self.leg = 0
        self.base_cmd = None
        self.hist = []
        self.push_start = None
        self.passes = 0
        self.phase_t = 0

    def _target_cube(self, cubes):
        while self.idx < len(self.order):
            name = self.order[self.idx]
            if name in cubes and cubes[name][1] > Y_GOAL + 0.015:
                return name
            self._next_cube()
        return None

    def _next_cube(self):
        self.phase_t = 0
        self.idx += 1
        self.bias_i = 0
        self.phase = "goto"
        self.leg = 0
        self.hist = []
        self.push_start = None
        self.base_cmd = None

    def _retry(self):
        self.phase_t = 0
        self.bias_i += 1
        self.phase = "goto"
        self.leg = 0
        self.hist = []
        self.push_start = None
        self.base_cmd = None
        if self.bias_i >= len(BIASES):
            self._next_cube()

    def _action(self, j, b, tgt_b, q=None, grip=0.0):
        a = np.zeros(11, dtype=np.float32)
        if tgt_b is not None:
            a[0] = np.clip((tgt_b[0] - b[0]) / BASE_GAIN, -0.1, 0.1)
            a[1] = np.clip((tgt_b[1] - b[1]) / BASE_GAIN, -0.1, 0.1)
        a[2] = np.clip(_wrap(-np.pi / 2 - b[2]) / ROT_GAIN, -0.1, 0.1)
        qq = Q_PUSH if q is None else q
        a[3:10] = np.clip((qq - j) / ARM_GAIN, -0.1, 0.1)
        a[10] = grip
        return a

    def get_action(self, state):
        self.t += 1
        self.phase_t = getattr(self, "phase_t", 0) + 1
        if self.phase_t > 200:
            self._retry()
        j, b = self._robot(state)
        cubes = self._cubes(state)
        name = self._target_cube(cubes)
        if name is None:
            # nothing left in this pass: restart a pass (cubes may have shifted)
            self.passes += 1
            if self.passes < 6:
                self.order = sorted(cubes.keys(), key=lambda n: cubes[n][1])
                self.idx = 0
                self.bias_i = 0
                self.phase = "goto"
                self.leg = 0
                self.base_cmd = None
                name = self._target_cube(cubes)
            if name is None:
                return self._action(j, b, None)

        c = cubes[name]
        bias = BIASES[min(self.bias_i, len(BIASES) - 1)]
        lane_x = c[0] - CONTACT_LAT + bias
        stage_y = c[1] + CONTACT_FWD + 0.12

        if self.phase == "goto":
            # leg 0: back off in +y to the staging line, leg 1: slide to the lane
            if self.leg == 0:
                tgt = np.array([b[0], max(stage_y, b[1])])
                if b[1] >= stage_y - 0.02:
                    self.leg = 1
                    tgt = np.array([lane_x, stage_y])
            else:
                tgt = np.array([lane_x, stage_y])
                if abs(b[0] - lane_x) < 0.02 and abs(b[1] - stage_y) < 0.03:
                    q_err = float(np.max(np.abs(Q_PUSH - j)))
                    if q_err < 0.16 and abs(_wrap(b[2] + np.pi / 2)) < 0.05:
                        self.phase = "push"
                        self.phase_t = 0
                        self.base_cmd = np.array([lane_x, b[1]])
                        self.push_start = b[1]
                        self.hist = []
            return self._action(j, b, tgt)

        # pushing: advance the commanded base target slowly along -y
        if self.base_cmd is None:
            self.base_cmd = np.array([lane_x, b[1]])
        self.base_cmd[0] = lane_x
        if b[1] - self.base_cmd[1] < 0.06:
            self.base_cmd[1] -= PUSH_SPEED
        self.hist.append(c[1])
        if len(self.hist) > 18:
            self.hist.pop(0)
            travelled = self.push_start - b[1]
            if travelled > 0.18 and (self.hist[0] - self.hist[-1]) < 0.012:
                self._retry()
                return self._action(j, b, None)
        if b[1] < Y_GOAL + CONTACT_FWD - 0.08:
            self._next_cube()
        return self._action(j, b, self.base_cmd)
