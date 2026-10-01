import numpy as np


class GeneratedApproach:
    """Coordinate-feedback pick and place controller for the three cubes."""

    OBJECTS = (0, 54, 70)

    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.large_offsets = ((0.49, 0.01), (0.48, 0.0), (0.50, 0.02),
                              (0.47, 0.01), (0.51, 0.0))
        self.small_offsets = ((0.44, 0.0), (0.43, 0.0), (0.45, 0.01),
                              (0.44, -0.015), (0.44, 0.015))
        self.home = np.array([0.0, -0.3491, np.pi, -2.5482,
                              0.0, -0.8727, np.pi / 2], dtype=float)
        self.pick = self.home.copy()
        self.pick[1] = 1.65
        self.pick[3] = -1.46
        # This wrist angle makes the finger pair vertical rather than diagonal.
        self.pick[5] = -0.25
        self.carry = self.pick.copy()
        self.carry[1] = 0.80

    def reset(self, state, info):
        self.obj_num = 0
        self.retry = 0
        self.phase = "approach"
        self.age = 0
        self.ground_z = [float(state[i + 2]) for i in self.OBJECTS]
        # Seesaw bbox lengths are full extents: keep small cube centers safely
        # inside the +/-0.18 m ends while balancing them symmetrically.
        self.target_offsets = (0.0, 0.12, -0.12)

    def _configure_pick(self):
        if self.obj_num == 0:
            self.pick[1], self.pick[5] = 1.65, -0.25
            self.pick_offsets = self.large_offsets
        else:
            self.pick[1], self.pick[5] = 1.70, -0.30
            self.pick_offsets = self.small_offsets

    @staticmethod
    def _clip(x, limit=0.1):
        return float(np.clip(x, -limit, limit))

    def _joint_control(self, state, target):
        a = np.zeros(11, dtype=np.float32)
        q = np.asarray(state[19:26], dtype=float)
        a[3:10] = np.clip(1.8 * (target - q), -0.1, 0.1)
        return a

    def _base_control(self, state, x, y):
        return (self._clip(0.9 * (x - float(state[16]))),
                self._clip(0.9 * (y - float(state[17]))),
                self._clip(-0.8 * float(state[18])))

    def _advance(self, phase):
        self.phase = phase
        self.age = 0

    def get_action(self, state):
        state = np.asarray(state)
        self.age += 1
        oi = self.OBJECTS[self.obj_num]
        cube = state[oi:oi + 3]
        seesaw = state[38:41]

        if self.phase == "approach":
            self._configure_pick()
            a = self._joint_control(state, self.pick)
            dx, dy = self.pick_offsets[self.retry]
            bx, by, br = self._base_control(state, cube[0] - dx, cube[1] - dy)
            a[:3] = (bx, by, br)
            a[10] = 0.0
            qerr = np.max(np.abs(state[19:26] - self.pick))
            berr = np.hypot(state[16] - (cube[0] - dx), state[17] - (cube[1] - dy))
            if (qerr < 0.035 and berr < 0.018 and self.age > 8) or self.age > 115:
                self._advance("close")
            return a

        if self.phase == "close":
            a = self._joint_control(state, self.pick)
            a[10] = 1.0
            if self.age >= 8:
                self._advance("lift")
            return a

        if self.phase == "lift":
            # A slow shoulder lift avoids batting the cube out of the pinch.
            a = self._joint_control(state, self.pick)
            a[4] = -0.025
            a[10] = 1.0
            if cube[2] > max(0.115, self.ground_z[self.obj_num] + 0.07):
                self.carry[1] = float(state[20])
                self._advance("carry")
            elif self.age >= 55:
                if self.retry + 1 < len(self.pick_offsets):
                    self.retry += 1
                    self._advance("approach")
                else:
                    self.retry = 0
                    self.obj_num = (self.obj_num + 1) % 3
                    self._advance("approach")
            return a

        yaw = 2.0 * np.arctan2(float(state[44]), float(state[41]))
        along = self.target_offsets[self.obj_num]
        tx = float(seesaw[0] + along * np.cos(yaw))
        ty = float(seesaw[1] + along * np.sin(yaw))
        target_z = float(seesaw[2] + state[53] + state[oi + 15])

        if self.phase == "carry":
            a = self._joint_control(state, self.carry)
            a[0] = self._clip(0.9 * (tx - cube[0]))
            a[1] = self._clip(0.9 * (ty - cube[1]))
            a[2] = self._clip(-0.8 * state[18])
            a[10] = 1.0
            if np.hypot(cube[0] - tx, cube[1] - ty) < 0.025 or self.age > 90:
                self._advance("lower")
            return a

        if self.phase == "lower":
            a = self._joint_control(state, self.carry)
            a[0] = self._clip(0.9 * (tx - cube[0]))
            a[1] = self._clip(0.9 * (ty - cube[1]))
            a[2] = self._clip(-0.8 * state[18])
            a[4] = self._clip(0.7 * (cube[2] - target_z), 0.06)
            a[10] = 1.0
            if cube[2] <= target_z + 0.012 or self.age > 60:
                self._advance("release")
            return a

        if self.phase == "release":
            # Open without moving the wrist; descending during release can press a
            # small cube through or off the narrow plank.
            a = np.zeros(11, dtype=np.float32)
            a[10] = 0.0
            if self.age >= 8:
                self._advance("retract")
            return a

        a = self._joint_control(state, self.carry)
        a[10] = 0.0
        if self.age >= 28:
            self.obj_num += 1
            self.retry = 0
            if self.obj_num >= 3:
                self.obj_num = 2
                self._advance("done")
            else:
                self._advance("approach")
        if self.phase == "done":
            a[:] = 0.0
        return a
