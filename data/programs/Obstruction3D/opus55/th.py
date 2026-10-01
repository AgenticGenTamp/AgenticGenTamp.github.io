from env_client import make_env
from kin import *
import numpy as np
GF=['grasp_tf_x','grasp_tf_y','grasp_tf_z','grasp_tf_qx','grasp_tf_qy','grasp_tf_qz','grasp_tf_qw']
BF=['pose_x','pose_y','pose_z','pose_qx','pose_qy','pose_qz','pose_qw']
class H:
    def __init__(self, seed, env=None, oc=None):
        self.env = env or make_env(); self.obs,_ = self.env.reset(seed=seed, options={'object_count':oc} if oc is not None else None)
        self.R = self.obs.get_object_from_name('robot'); self.n=0; self.done=False
    def o(self, name): return self.obs.get_object_from_name(name)
    def g(self, name, f): return self.obs.get(self.o(name), f)
    def q(self): return np.array([self.obs.get(self.R,f'joint_{i}') for i in range(1,8)])
    def base(self): return tuple(self.obs.get(self.R,f) for f in ('pos_base_x','pos_base_y','pos_base_rot'))
    def tool(self): return fk(self.base(), self.q())
    def step(self, a):
        self.obs, r, te, tr, _ = self.env.step(np.asarray(a,dtype=np.float32)); self.n+=1
        if te: self.done=True
        return te
    def grip(self, v):
        a=np.zeros(11); a[10]=v; return self.step(a)
    def goto_q(self, qt, grip=0.0, maxsteps=60):
        for _ in range(maxsteps):
            q=self.q(); d=qt-q
            if np.max(np.abs(d))<1e-5: return True
            a=np.zeros(11); a[3:10]=np.clip(d,-0.2,0.2); a[10]=grip
            self.step(a)
            if np.allclose(self.q(),q): return False
        return False
    def goto(self, pos, yaw=0.0):
        qt,e=ik(self.base(), self.q(), np.asarray(pos,float), down_R(yaw))
        return self.goto_q(qt), e
    def grasped(self): return self.obs.get(self.R,'grasp_active')>0.5
    def pose(self, name): return np.array([self.g(name,f) for f in BF])
    def he(self, name): return np.array([self.g(name,'half_extent_'+c) for c in 'xyz'])
