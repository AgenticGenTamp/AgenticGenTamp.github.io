import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, fk, ctrl

class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        pass
    def reset(self, state, info):
        self.t = 0
        self.q = None
    def get_action(self, state):
        r = state.get_object_from_name("robot")
        J = np.array([float(state.get(r, "pos_arm_joint%d" % i)) for i in range(1, 8)])
        B = np.array([float(state.get(r, f)) for f in ("pos_base_x","pos_base_y","pos_base_rot")])
        c = state.get_object_from_name("cube_0")
        C = np.array([float(state.get(c, f)) for f in ("x","y")])
        yaw = B[2]; fwd = np.array([np.cos(yaw), np.sin(yaw)])
        self.t += 1
        if self.t < 100:
            q = ctrl.ik_local([0.45, 0.0, 0.18], J)
            return ctrl.action(J, q, B, None, 0.0)
        q = ctrl.ik_local([0.45, 0.0, 0.18 if self.t < 190 else -0.02], J)
        p, _ = fk.fk(J, *B)
        d = p[:2] - B[:2]
        want = C - 0.15 * fwd
        tb = np.array([want[0]-d[0], want[1]-d[1], yaw])
        return ctrl.action(J, q, B, tb, 0.0)
