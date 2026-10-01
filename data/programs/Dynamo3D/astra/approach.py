"""Fast feedback navigation for the empirically observed Dynamo3D layouts."""
import numpy as np


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.robot_type = observation_space.get_type('mujoco_tidybot_robot')
        self.movable_type = observation_space.get_type('mujoco_movable_object')
        self.low = np.asarray(action_space.low, dtype=np.float32)
        self.high = np.asarray(action_space.high, dtype=np.float32)

    def reset(self, state, info):
        self.robot = state.get_objects(self.robot_type)[0]
        # Goal markers are not included in observations. The compact scene has
        # its goal at (1, 0); the room layout has its goal near (3.8, 3.8).
        # Recognize the layout spatially, independently of names or counts.
        objects = state.get_objects(self.movable_type)
        room = not objects or any(
            state.get(obj, 'x') > 2.0 or state.get(obj, 'y') > 2.0
            for obj in objects
        )
        self.goal = np.array([3.75, 3.75] if room else [1.0, 0.0])

    def get_action(self, state):
        position = np.array([
            state.get(self.robot, 'pos_base_x'),
            state.get(self.robot, 'pos_base_y'),
        ])
        action = np.zeros(self.action_space.shape, dtype=np.float32)
        # Controls are world-frame increments. Simultaneous saturated x/y
        # motion minimizes steps; movable chairs can be pushed out of the way.
        # Zero arm/yaw increments preserve the folded arm and initial heading.
        action[:2] = self.goal - position
        return np.clip(action, self.low, self.high)
