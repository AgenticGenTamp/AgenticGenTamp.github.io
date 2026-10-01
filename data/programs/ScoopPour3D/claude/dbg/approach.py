import numpy as np, kin, json, os
class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives=None):
        self.aspace = action_space
        cfg = json.load(open(os.path.join(os.path.dirname(__file__),'cfg.json')))
        self.targets = cfg['targets']  # list of [x,y,z,hold]
        self.tool = cfg.get('tool',0.12)
    def reset(self, state, info):
        self.t = 0; self.idx = 0; self.hold = 0
        r = state.get_object_from_name('robot')
        self.R = r
        self.base = state.data[r][:3].copy()
        self.q = state.data[r][3:10].copy()
        self.qd = self.q.copy()
        self._plan(state)
    def _plan(self, state):
        tx,ty,tz,hold = self.targets[min(self.idx,len(self.targets)-1)]
        self.qd,err = kin.ik(np.array([tx,ty,tz]), kin.rot_down(), self.q, self.tool)
        self.holdn = hold
    def get_action(self, state):
        q = state.data[self.R][3:10]
        a = np.zeros(11, dtype=np.float32)
        a[3:10] = np.clip(self.qd - q, -0.1, 0.1)
        a[10] = 0.0
        self.hold += 1
        if (np.abs(self.qd-q).max() < 0.01 or self.hold > self.holdn) and self.idx < len(self.targets)-1:
            self.idx += 1; self.q = q.copy(); self.hold=0; self._plan(state)
        return a
