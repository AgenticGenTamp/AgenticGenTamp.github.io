"""Candidate wrapper adding a safe right-boundary-to-left pick/place rule.

Kept separate from approach.py so the parent agent can review/integrate the rule.
"""
import math

import numpy as np

from approach import GeneratedApproach as BaselineApproach


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.low = np.asarray(action_space.low, dtype=float)
        self.high = np.asarray(action_space.high, dtype=float)
        self.robot_type = observation_space.get_type("kin_robot")
        self.block_type = observation_space.get_type("target_block")
        self.surface_type = observation_space.get_type("target_surface")
        self.baseline = BaselineApproach(action_space, observation_space, primitives)
        self.reset(None, None)

    def reset(self, state, info):
        self.baseline.reset(state, info)
        self.special = False
        self.phase = "raise"
        self.phase_step = 0
        if state is not None:
            robot = state.get_objects(self.robot_type)[0]
            block = state.get_objects(self.block_type)[0]
            surface = state.get_objects(self.surface_type)[0]
            get = lambda obj, feature: float(state.get(obj, feature))
            obstructed = any(name.startswith("obstruction")
                             for name in state.get_object_names())
            # The baseline's behind-object staging point is outside the right
            # wall here.  A top-down grasp is possible only if the cargo fits
            # inside the fully open finger gap.
            self.special = (not obstructed and get(surface, "x") < get(block, "x")
                            and get(block, "x") > 2.70
                            and get(block, "width") < get(robot, "finger_gap") - .005)

    @staticmethod
    def _angle_error(want, have):
        return (want - have + math.pi) % (2.0 * math.pi) - math.pi

    def _advance(self, phase):
        self.phase = phase
        self.phase_step = 0

    def get_action(self, state):
        if not self.special:
            return self.baseline.get_action(state)

        robot = state.get_objects(self.robot_type)[0]
        block = state.get_objects(self.block_type)[0]
        surface = state.get_objects(self.surface_type)[0]
        get = lambda obj, feature: float(state.get(obj, feature))
        rx, ry = get(robot, "x"), get(robot, "y")
        bx, by = get(block, "x"), get(block, "y")
        action = np.zeros(5, dtype=float)

        if self.phase == "raise":
            action[1] = np.clip(1.12 - ry, -.049, .049)
            action[3], action[4] = -.099, .019
            if abs(ry - 1.12) < .012 or self.phase_step >= 30:
                self._advance("align")
        elif self.phase == "align":
            error = self._angle_error(-math.pi / 2.0, get(robot, "theta"))
            action[0] = np.clip(bx - rx, -.049, .049)
            action[2] = np.clip(error, -.19, .19)
            action[3], action[4] = -.099, .019
            if ((abs(bx-rx) < .01 and abs(error) < .02)
                    or self.phase_step >= 70):
                self._advance("lower")
        elif self.phase == "lower":
            target_y = by + .40
            action[1] = np.clip(target_y - ry, -.049, .049)
            action[3], action[4] = -.099, .019
            if abs(target_y-ry) < .01 or self.phase_step >= 35:
                self._advance("close")
        elif self.phase == "close":
            action[4] = -.019
            if get(block, "held") > .5 or self.phase_step >= 14:
                self._advance("lift")
        elif self.phase == "lift":
            action[1] = np.clip(.62 - by, -.035, .035)
            if abs(by-.62) < .012 or self.phase_step >= 30:
                self._advance("translate")
        elif self.phase == "translate":
            target_x = get(surface, "x")
            action[0] = np.clip(target_x - bx, -.025, .025)
            if abs(target_x-bx) < .008 or self.phase_step >= 45:
                self._advance("place")
        elif self.phase == "place":
            target_y = (get(surface, "y") + get(surface, "height") / 2.0
                        + get(block, "height") / 2.0 + .006)
            action[1] = np.clip(target_y - by, -.025, .025)
            if abs(target_y-by) < .008 or self.phase_step >= 30:
                self._advance("release")
        elif self.phase == "release":
            action[4] = .019
            if get(block, "held") < .5 or self.phase_step >= 14:
                self._advance("withdraw")
        else:
            action[1] = .035

        self.phase_step += 1
        return np.clip(action, self.low + 1e-7, self.high - 1e-7).astype(float)
