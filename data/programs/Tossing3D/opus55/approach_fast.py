"""Tossing3D approach: pick each cube and throw it over the barrier into the bin.

Logic lives in toss.py (generator policy) and kin.py (FK/IK for Kinova Gen3 on tidybot).
"""
import numpy as np
import toss_fast as toss


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self._gen = None
        self._started = False

    def reset(self, state, info):
        self._gen = toss.toss_all(state)
        self._started = False

    def _safe(self, a):
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        if a.shape[0] != 18 or not np.all(np.isfinite(a)):
            a = np.zeros(18, dtype=np.float32)
        lo = np.asarray(self.action_space.low, dtype=np.float32)
        hi = np.asarray(self.action_space.high, dtype=np.float32)
        return np.clip(a, lo, hi).astype(np.float32)

    def get_action(self, state):
        if self._gen is None:
            self.reset(state, {})
        try:
            if not self._started:
                self._started = True
                a = next(self._gen)
            else:
                a = self._gen.send(state)
        except StopIteration:
            a = np.zeros(18, dtype=np.float32)
        except Exception:
            # restart the plan from the current state on any unexpected error
            self._gen = toss.toss_all(state)
            try:
                a = next(self._gen)
            except Exception:
                a = np.zeros(18, dtype=np.float32)
        return self._safe(a)
