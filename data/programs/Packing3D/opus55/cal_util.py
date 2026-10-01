import numpy as np
from env_client import make_env
RF=['pos_base_x','pos_base_y','pos_base_rot']
JF=[f'joint_{i}' for i in range(1,8)]
GF=['grasp_tf_x','grasp_tf_y','grasp_tf_z','grasp_tf_qx','grasp_tf_qy','grasp_tf_qz','grasp_tf_qw']
PF=['pose_x','pose_y','pose_z','pose_qx','pose_qy','pose_qz','pose_qw']
def rob(obs):
    R=obs.get_object_from_name('robot')
    return dict(base=np.array([obs.get(R,f) for f in RF]), q=np.array([obs.get(R,f) for f in JF]),
                finger=obs.get(R,'finger_state'), ga=obs.get(R,'grasp_active'), gtf=np.array([obs.get(R,f) for f in GF]))
def part(obs,name):
    P=obs.get_object_from_name(name)
    return np.array([obs.get(P,f) for f in PF]), obs.get(P,'grasp_active')
def act(dq=np.zeros(7), dbase=np.zeros(3), grip=0.0):
    a=np.zeros(11,dtype=np.float32); a[:3]=dbase; a[3:10]=dq; a[10]=grip; return a

class Driver:
    def __init__(self, seed=0):
        self.env=make_env(); self.obs,_=self.env.reset(seed=seed); self.nrej=0; self.steps=0; self.A=[]
    @property
    def r(self): return rob(self.obs)
    def step(self, a):
        before=rob(self.obs)
        self.obs,rew,term,trunc,info=self.env.step(a); self.steps+=1; self.A.append(np.array(a))
        after=rob(self.obs)
        moved=np.abs(np.r_[after['q']-before['q'],after['base']-before['base']]).max()
        want=np.abs(a[:10]).max()
        rej = want>1e-6 and moved<1e-9
        if rej: self.nrej+=1
        return rej
    def goto_q(self, qt, max_step=0.19, grip=0.0):
        from ik import wrap_near
        q=self.r['q']; qt=wrap_near(qt,q)
        d=qt-q; n=max(1,int(np.ceil(np.abs(d).max()/max_step)))
        for i in range(n):
            q=self.r['q']; rem=qt-q; s=np.clip(rem/(n-i),-0.2,0.2)
            if self.step(act(dq=s,grip=grip)): return False
        return True
    def goto_base(self, bt, max_step=0.19):
        b=self.r['base']; d=np.asarray(bt)-b; n=max(1,int(np.ceil(np.abs(d).max()/max_step)))
        for i in range(n):
            b=self.r['base']; s=np.clip((np.asarray(bt)-b)/(n-i),-0.2,0.2)
            if self.step(act(dbase=s)): return False
        return True

    def save(self, path='cal_render/actions.npy'):
        np.save(path, np.array(self.A,dtype=np.float32))
