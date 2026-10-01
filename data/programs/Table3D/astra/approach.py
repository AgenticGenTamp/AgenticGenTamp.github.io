"""Fast, observation-driven pick-and-lift controller.

The two downward-facing arm postures were measured against the live robot.
Selecting the leftmost cube leaves a clear approach from the table's near edge.
"""
import numpy as np


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.rt = observation_space.get_type('Kinematic3DRobot')
        self.ct = observation_space.get_type('Kinematic3DCuboid')
        self.fields = ['pos_base_x', 'pos_base_y', 'pos_base_rot'] + [
            'joint_' + str(i) for i in range(1, 8)
        ]
        self.low = np.asarray(action_space.low)
        self.high = np.asarray(action_space.high)

    def reset(self, state, info):
        self.t = 0
        robots = list(state.get_objects(self.rt))
        self.r = robots[0] if robots else None
        cubes = [o for o in state.get_objects(self.ct) if o.name != 'table']
        self.cubes = sorted(cubes, key=lambda o: state.get(o, 'pose_x'))
        self.candidate_index = 0
        self.attempt_step = 0
        self.round = 0
        self.c = self.cubes[0] if self.cubes else None

    def get_action(self, state):
        self.t += 1
        action = np.zeros(11, dtype=np.float32)
        if self.c is None or self.r is None:
            return action
        action[10] = -1
        if state.get(self.r, 'grasp_active'):
            # Raising the shoulder lifts the held cube in one action.
            action[4] = self.low[4]
            return action

        self.attempt_step += 1
        if self.attempt_step in (13, 14):
            # Clear the surrounding cubes before changing targets.
            action[4] = max(-.3, float(self.low[4]))
            return action
        if self.attempt_step > 14:
            self.candidate_index = (self.candidate_index + 1) % len(self.cubes)
            if self.candidate_index == 0:
                self.round += 1
            self.c = self.cubes[self.candidate_index]
            self.attempt_step = 1

        x = state.get(self.c, 'pose_x')
        y = state.get(self.c, 'pose_y')
        if x < .65 and self.round % 2 == 0:
            q2, q4, reach = .434994191, -2.3, .44153585
        else:
            q2, q4, reach = .69412772, -1.7, .62276276
        base_x, base_y = x - reach, y - .00135
        # Lateral retries avoid fingers striking a neighboring cube.
        if 5 <= self.attempt_step <= 6:
            base_y += .02
        elif 7 <= self.attempt_step <= 8:
            base_y -= .02
        elif 9 <= self.attempt_step <= 10:
            base_x -= .015
        elif 11 <= self.attempt_step <= 12:
            base_x += .015
        q6 = q2 - q4 - np.pi
        target = np.array([
            base_x, base_y, 0, 0, q2, -np.pi, q4, 0, q6, np.pi / 2
        ])
        current = np.array([state.get(self.r, f) for f in self.fields])
        action[:10] = np.clip(target - current, self.low[:10], self.high[:10])
        return action
