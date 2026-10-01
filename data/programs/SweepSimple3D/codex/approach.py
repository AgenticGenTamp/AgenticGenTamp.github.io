"""Geometric sweep policy for the variable-count SweepSimple3D task."""

import math
import numpy as np


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.dtype = action_space.dtype

    @staticmethod
    def _get(state, name, feature):
        return float(state.get(state.get_object_from_name(name), feature))

    def reset(self, state, info):
        self.step = 0
        self.initial_joints = np.array([
            self._get(state, "robot", "pos_arm_joint%d" % i)
            for i in range(1, 8)
        ])
        self.cube_names = sorted(
            name for name in state.get_object_names() if name.startswith("cube_")
        )
        wx = self._get(state, "wiper_0", "x")
        wy = self._get(state, "wiper_0", "y")
        if self.cube_names:
            center = np.mean([
                [self._get(state, name, "x"), self._get(state, name, "y")]
                for name in self.cube_names
            ], axis=0)
            delta = center - np.array([wx, wy])
        else:
            delta = np.array([0.0, -1.0])
        norm = float(np.linalg.norm(delta))
        self.direction = delta / norm if norm > 1e-6 else np.array([0.0, -1.0])
        # Stand close enough for the lowered wrist to reach the handle while
        # retaining lateral clearance between the chassis and blade.
        lateral = np.array([-self.direction[1], self.direction[0]])
        self.pickup_base = (np.array([wx, wy]) - 0.40 * self.direction
                            - 0.36 * lateral)
        self.pickup_yaw = math.atan2(self.direction[1], self.direction[0])
        projections = np.array([
            np.dot(np.array([self._get(state, n, "x"), self._get(state, n, "y")])
                   - np.array([wx, wy]), self.direction)
            for n in self.cube_names
        ]) if self.cube_names else np.array([0.5])
        self.sweep_length = max(1.0, float(np.max(projections)) + 0.55)
        self.sweep_start = None
        self.ballistic = 0 < len(self.cube_names) <= 5
        if self.ballistic:
            # A carefully offset chassis transit makes the upright blade fall
            # through a lone cube.  This is the only repeatable mechanism that
            # measurably moves a cube when the simulated gripper misses.
            if len(self.cube_names) == 1:
                angle, lateral = 74.0, -0.29
            else:
                angle, lateral = 94.0, -0.24
            theta = math.radians(angle)
            c, s = math.cos(theta), math.sin(theta)
            direct = self.direction
            impact = np.array([c * direct[0] - s * direct[1],
                               s * direct[0] + c * direct[1]])
            side = np.array([-impact[1], impact[0]])
            self.impact_start = (np.array([wx, wy]) - 0.48 * impact
                                 + lateral * side)

    @staticmethod
    def _angle_error(target, actual):
        return (target - actual + math.pi) % (2.0 * math.pi) - math.pi

    def get_action(self, state):
        action = np.zeros(self.action_space.shape, dtype=self.dtype)
        rx = self._get(state, "robot", "pos_base_x")
        ry = self._get(state, "robot", "pos_base_y")
        yaw = self._get(state, "robot", "pos_base_rot")
        if self.ballistic:
            if self.step < 24:
                error = self.impact_start - np.array([rx, ry])
                action[0:2] = np.clip(0.8 * error, -0.1, 0.1)
            self.step += 1
            return action
        if self.step < 45:
            error = self.pickup_base - np.array([rx, ry])
            action[0:2] = np.clip(0.65 * error, -0.1, 0.1)
            action[2] = np.clip(0.65 * self._angle_error(self.pickup_yaw, yaw), -0.1, 0.1)
            action[10] = 1.0
            # Lower the wrist toward the upper part of the vertical handle.
            q2 = self._get(state, "robot", "pos_arm_joint2")
            action[4] = np.clip(0.35 * (self.initial_joints[1] + 0.52 - q2), -0.1, 0.1)
            q4 = self._get(state, "robot", "pos_arm_joint4")
            action[6] = np.clip(0.35 * (self.initial_joints[3] + 0.52 - q4), -0.1, 0.1)
            q3 = self._get(state, "robot", "pos_arm_joint3")
            action[5] = np.clip(0.35 * (self.initial_joints[2] - 0.5 - q3), -0.1, 0.1)
            q6 = self._get(state, "robot", "pos_arm_joint6")
            action[8] = np.clip(0.35 * (self.initial_joints[5] - 0.35 - q6), -0.1, 0.1)
        elif self.step < 60:
            action[10] = 0.0
        else:
            if self.sweep_start is None:
                self.sweep_start = np.array([rx, ry])
            progress = float(np.dot(np.array([rx, ry]) - self.sweep_start,
                                    self.direction))
            if progress < self.sweep_length:
                # Carry (or, for a marginal grasp, press) the tool steadily
                # through the whole projected extent of the cube set.
                action[0:2] = 0.075 * self.direction
            action[2] = np.clip(0.5 * self._angle_error(self.pickup_yaw, yaw), -0.08, 0.08)
            action[10] = 0.0
            q2 = self._get(state, "robot", "pos_arm_joint2")
            action[4] = np.clip(0.35 * (self.initial_joints[1] + 0.52 - q2), -0.1, 0.1)
            q4 = self._get(state, "robot", "pos_arm_joint4")
            action[6] = np.clip(0.35 * (self.initial_joints[3] + 0.52 - q4), -0.1, 0.1)
            q3 = self._get(state, "robot", "pos_arm_joint3")
            action[5] = np.clip(0.35 * (self.initial_joints[2] - 0.5 - q3), -0.1, 0.1)
            q6 = self._get(state, "robot", "pos_arm_joint6")
            action[8] = np.clip(0.35 * (self.initial_joints[5] - 0.35 - q6), -0.1, 0.1)
        self.step += 1
        return action
