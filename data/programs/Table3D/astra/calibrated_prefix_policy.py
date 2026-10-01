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
        self.c = min(cubes, key=lambda o: state.get(o, 'pose_x')) if cubes else None

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

        x = state.get(self.c, 'pose_x')
        y = state.get(self.c, 'pose_y')
        if self.t <= 12:
            if x < .65:
                q2, q4, reach = .434994191, -2.3, .44153585
            else:
                q2, q4, reach = .69412772, -1.7, .62276276
            # A small offset toward the robot avoids adjacent cubes.
            # If contact blocks the descent, try nearby grasp positions.
            if self.t > 4:
                reach += (.015, -.015, .03, -.03)[((self.t - 5) // 2) % 4]
            base_x, base_y = x - reach, y - .00135
        else:
            # Recovery search changes both extension and approach position.
            n = (self.t - 12) // 16
            q2 = -.9 + .2 * (n % 10)
            q4 = -2.7 + .3 * ((n // 10) % 5)
            base_x, base_y = x - .65 + .05 * (self.t % 16), y
        q6 = q2 - q4 - np.pi
        target = np.array([
            base_x, base_y, 0, 0, q2, -np.pi, q4, 0, q6, np.pi / 2
        ])
        current = np.array([state.get(self.r, f) for f in self.fields])
        action[:10] = np.clip(target - current, self.low[:10], self.high[:10])
        return action
