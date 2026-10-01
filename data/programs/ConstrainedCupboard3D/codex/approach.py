"""Bounded feedback policy for the variable-count cupboard task."""
import math
import numpy as np


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self.home = np.array([0., -.3490659, np.pi, -2.5481806,
                              0., -.8726646, np.pi / 2])
        # Empirically lowest valid configuration from a rendered joint-space
        # search (the fingertips reach the rod plane without hitting a stop).
        # Corrected public-URDF solution whose downward fingers repeatedly
        # produced measured rod contact in the live simulator.
        self.pick_q = np.array([-3.0175, -2.3524, -1.1879, -.314,
                                -1.5966, .6707, -2.7101])

    def reset(self, state, info):
        mov = self.observation_space.get_type("mujoco_movable_object")
        self.rods = sorted(list(state.get_objects(mov)),
                           key=lambda o: (float(state.get(o, "y")), o.name))
        fix = self.observation_space.get_type("mujoco_fixture")
        ys = sorted(float(state.get(o, "y")) for o in state.get_objects(fix))
        n = len(self.rods)
        if n and len(ys) >= 2:
            gaps = np.array([(a + b) / 2 for a, b in zip(ys[:-1], ys[1:])])
            # Rods are interchangeable and several layouts expose spare gaps.
            # Assign each sorted rod its nearest unused gap to minimize the
            # difficult lateral correction required from the floor pusher.
            available = list(gaps)
            chosen = []
            for rod in self.rods:
                value = float(state.get(rod, "y"))
                if not available:
                    chosen.append(float(np.clip(value, gaps[0], gaps[-1])))
                    continue
                j = min(range(len(available)), key=lambda k: abs(available[k] - value))
                chosen.append(available.pop(j))
            self.slot_y = chosen
        else:
            self.slot_y = [0.] * n
        self.initial = {o.name: np.array([float(state.get(o, f))
                                         for f in ("x", "y", "z")])
                        for o in self.rods}
        self.push_side = {o.name: (.10 if self.initial[o.name][1] < self.slot_y[i]
                                   else -.10)
                          for i, o in enumerate(self.rods)}
        self.step = 0

    @staticmethod
    def _v(state, obj, feature):
        return float(state.get(obj, feature))

    def _joints(self, state, robot, action, target):
        q = np.array([self._v(state, robot, f"pos_arm_joint{i}")
                      for i in range(1, 8)])
        err = (target - q + np.pi) % (2 * np.pi) - np.pi
        action[3:10] = np.clip(.5 * err, -.1, .1)

    def _base(self, state, robot, action, x, y, yaw=None):
        action[0] = np.clip((x - self._v(state, robot, "pos_base_x")) / .87, -.1, .1)
        action[1] = np.clip((y - self._v(state, robot, "pos_base_y")) / .87, -.1, .1)
        if yaw is not None:
            err = (yaw - self._v(state, robot, "pos_base_rot") + np.pi) % (2*np.pi) - np.pi
            action[2] = np.clip(err, -.1, .1)

    def get_action(self, state):
        action = np.zeros(self.action_space.shape, dtype=self.action_space.dtype)
        if not self.rods:
            return action
        # First unfold the arm while the base is clear of every object.  Moving
        # the base simultaneously changes the collision branch and prevents
        # this low configuration from being reached.
        if self.step < 120:
            self.step += 1
            action[-1] = 1.0
            robot = state.get_object_from_name("robot")
            first = self.rods[0]
            fp = self.initial[first.name]
            self._base(state, robot, action, fp[0] - .80, fp[1] - .50, 0.)
            self._joints(state, robot, action, self.pick_q)
            return action
        cycle = 62
        repeats = max(1, min(7, 12 // len(self.rods)))
        stroke = (self.step - 120) // cycle
        index = stroke // repeats
        if index >= len(self.rods):
            action[-1] = 1.0
            return action
        phase = (self.step - 120) % cycle
        self.step += 1
        rod = self.rods[index]
        robot = state.get_object_from_name("robot")
        rx = self._v(state, rod, "x")
        ry = self._v(state, rod, "y")
        if rx >= 1.75:
            action[-1] = 1.0
            self._base(state, robot, action, rx - 1.0, ry, 0.)
            self._joints(state, robot, action, self.pick_q)
            return action
        side = .10 if ry < self.slot_y[index] else -.10
        if phase == 0:
            self.stroke_x = rx
        if phase < 25:
            action[-1] = 1.
            self._base(state, robot, action, rx - .80, ry - 5 * side, 0.)
            self._joints(state, robot, action, self.pick_q)
        elif phase < 37:
            action[-1] = 1.
            self._base(state, robot, action, rx - .80, ry + side, 0.)
            self._joints(state, robot, action, self.pick_q)
        elif phase < 47:
            self._base(state, robot, action, rx - .80, ry + side, 0.)
            self._joints(state, robot, action, self.pick_q)
        else:
            goal_x = min(1.60, self.stroke_x + .80)
            self._base(state, robot, action, goal_x, ry + side, 0.)
            self._joints(state, robot, action, self.pick_q)
        return action
