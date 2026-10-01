import numpy as np
from grasp import HookGrasp
from pull import HookPull

class GeneratedApproach:
    """Acquire the hook, lift it around the target, and pull onto the wall."""
    def __init__(self, action_space, observation_space, primitives):
        self.hook_type = observation_space.get_type('hook')
        self.grasp = HookGrasp(action_space, observation_space)
        self.pull = HookPull(action_space, observation_space)
        self.low = np.asarray(action_space.low) + 1e-7
        self.high = np.asarray(action_space.high) - 1e-7
    def reset(self, state, info):
        self.grasp.reset(state)
        self.pull.reset(state)
        self.held = False
        self.phase = 'grasp'
    def get_action(self, state):
        hooks = state.get_objects(self.hook_type)
        held = any(state.get(h, 'held') for h in hooks)
        if held:
            if not self.held:
                self.pull.reset(state)
            self.phase = 'pull'
            action = self.pull.get_action(state)
        else:
            if self.held:
                self.grasp.reset(state)
            self.phase = 'grasp'
            action = self.grasp.get_action(state)
        self.held = held
        return np.clip(action, self.low, self.high)
