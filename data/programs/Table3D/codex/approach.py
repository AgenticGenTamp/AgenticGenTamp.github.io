"""Fast feedback policy for variable-count Table3D cube pickup."""

import numpy as np


class GeneratedApproach:
    """Align the mobile base, move through calibrated top-down grasps, and lift."""

    # (world reach x, joint-2 offset, joint-4 offset, joint-6 offset).
    REACHES = (
        (0.431, 0.775, 0.000, 0.775),
        (0.516, 0.963, 0.000, 1.270),
        (0.556, 0.900, 0.000, 1.337),
        (0.604, 1.117, 0.000, 1.730),
        (0.637, 1.623, 0.070, 2.662),
        (0.696, 1.249, 0.225, 1.867),
        (0.757, 1.416, 0.498, 1.966),
    )

    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.low = np.asarray(action_space.low, dtype=np.float32)
        self.high = np.asarray(action_space.high, dtype=np.float32)
        self.robot_type = observation_space.get_type("Kinematic3DRobot")
        self.cuboid_type = observation_space.get_type("Kinematic3DCuboid")

    def _objects(self, state):
        robot = state.get_objects(self.robot_type)[0]
        cubes = [obj for obj in state.get_objects(self.cuboid_type)
                 if obj.name.startswith("cube")]
        return robot, cubes

    def reset(self, state, info):
        robot, cubes = self._objects(state)
        self.home = np.array(
            [state.get(robot, "joint_%d" % i) for i in range(1, 8)],
            dtype=np.float32,
        )
        jobs = []
        for cube in cubes:
            x = state.get(cube, "pose_x")
            y = state.get(cube, "pose_y")
            for reach in self.REACHES:
                # Prefer the closest cube: far-reach elbow poses can make a
                # later return to a near pose collision-blocked.
                jobs.append((x, abs(x - reach[0]), cube.name, y, reach))
        jobs.sort(key=lambda item: (item[0], item[1]))
        self.jobs = jobs
        self.job_index = 0
        self.phase_steps = 0
        self.extending = False

    def get_action(self, state):
        robot, _ = self._objects(state)
        action = np.zeros(self.action_space.shape, dtype=np.float32)
        if state.get(robot, "grasp_active") > 0.5:
            action[4] = -0.4
            action[10] = -1.0
            return np.clip(action, self.low, self.high).astype(np.float32)
        if not self.jobs:
            return action
        _, _, _, target_y, reach = self.jobs[self.job_index]
        current = np.array(
            [state.get(robot, "joint_%d" % i) for i in range(1, 8)],
            dtype=np.float32,
        )
        # Route around the pedestal: pitch the wrist while the shoulder is
        # retracted, then extend to the desired radial reach.
        target = self.home.copy()
        target[1] += 0.35 if not self.extending else reach[1]
        target[3] += reach[2]
        target[5] += reach[3]
        base_error = target_y - state.get(robot, "pos_base_y")
        reached = max(float(np.max(np.abs(target - current))), abs(base_error)) < 0.012
        limit = 8 if not self.extending else 6
        if reached or self.phase_steps >= limit:
            if not self.extending:
                self.extending = True
                self.phase_steps = 0
                target[1] = self.home[1] + reach[1]
            else:
                action[10] = -1.0
                self.job_index = (self.job_index + 1) % len(self.jobs)
                self.extending = False
                self.phase_steps = 0
                return action
        action[1] = np.clip(target_y - state.get(robot, "pos_base_y"), -0.4, 0.4)
        action[3:10] = np.clip(target - current, -0.4, 0.4)
        action[10] = 1.0
        self.phase_steps += 1
        return np.clip(action, self.low, self.high).astype(np.float32)
