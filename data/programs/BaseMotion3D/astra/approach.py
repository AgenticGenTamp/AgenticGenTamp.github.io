import numpy as np


class GeneratedApproach:
    """Use maximum base translation, with goal-region search at collisions."""

    def __init__(self, action_space, observation_space, primitives):
        self.low = np.asarray(action_space.low)
        self.high = np.asarray(action_space.high)
        self.dtype = action_space.dtype

    def reset(self, state, info):
        self.previous = None
        self.previous_action = None
        self.probe = -1
        self.angle = 0.0
        self.heading = 0.0
        self.offset = np.zeros(2)
        target = np.asarray(state)[19:21]
        if target[1] < -1.9:
            # Empirically measured wall front and its two exposed corners.
            wall_point = np.array([np.clip(target[0], -.4736748, 1.1421065), -2.163255])
            normal = target-wall_point
            distance = float(np.linalg.norm(normal))
            if 1e-8 < distance < .35:
                normal /= distance
                # The base's narrow side faces the nearest wall point.
                self.heading = float(np.arctan2(normal[1], normal[0])-np.pi/2)
                self.offset = .049999*normal

    def get_action(self, state):
        state = np.asarray(state)
        pos, target = state[:2], state[19:21]
        blocked = (self.previous is not None
                   and np.linalg.norm(pos - self.previous) < 1e-6
                   and np.linalg.norm(self.previous_action[:2]) > 1e-6)
        if blocked:
            if self.probe < 0:
                self.angle = np.pi/2 if target[1] < -1.9 else float(np.arctan2(pos[1]-target[1], pos[0]-target[0]))
                self.probe = 0
            else:
                self.probe += 1
        aim = target.copy()
        # The fixed room wall is south of the sampled target region.
        # Use the near edge of the goal disk there to avoid a rejected move.
        if target[1] < -1.9:
            aim[1] += min(.049999, -1.9-float(target[1]))
        if np.any(self.offset):
            aim = target + self.offset
        turn = (self.heading-state[2]+np.pi) % (2*np.pi)-np.pi
        if self.probe >= 0:
            # Collision tests reject a whole move. Try the goal disk's rim,
            # starting on the side facing the current, collision-free base.
            headings = (self.heading, -np.pi/4, np.pi/4, 1.09, np.pi/2, -np.pi/2, np.pi, 0.)
            if self.probe < 64:
                phase, k = divmod(self.probe, 8)
            else:
                phase, k = divmod(self.probe-64, 64)
            heading = headings[min(phase, len(headings)-1)]
            turn = (heading-state[2]+np.pi) % (2*np.pi)-np.pi
            if abs(turn) > 1e-5:
                action = np.zeros(11, dtype=self.dtype)
                action[2] = np.clip(turn, self.low[2], self.high[2])
                self.previous = pos.copy()
                self.previous_action = action.copy()
                return action
            # Bit reversal refines the full circle without duplicate probes.
            fraction, scale = 0.0, .5
            while k:
                fraction += (k & 1)*scale
                k >>= 1
                scale *= .5
            theta = self.angle + 2*np.pi*fraction
            aim = target + .049999*np.array([np.cos(theta), np.sin(theta)])
        action = np.zeros(11, dtype=self.dtype)
        action[:2] = np.clip(aim-pos, self.low[:2], self.high[:2])
        action[2] = np.clip(turn, self.low[2], self.high[2])
        self.previous = pos.copy()
        self.previous_action = action.copy()
        return action
