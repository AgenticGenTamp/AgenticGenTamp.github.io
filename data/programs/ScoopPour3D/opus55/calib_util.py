import numpy as np
from env_client import make_env
import kin
RD = np.array([[0,-1,0],[1,0,0],[0,0,1.]])
JN = ['pos_arm_joint%d'%i for i in range(1,8)]
class R:
    def __init__(s, seed=0):
        s.env = make_env(); s.obs,_ = s.env.reset(seed=seed)
        s.qi = s.q().copy(); s.g = 0.0; s.rew=[]
    def rob(s): return s.obs.get_object_from_name('robot')
    def q(s): return np.array([s.obs.get(s.rob(), j) for j in JN])
    def base(s): r=s.rob(); return np.array([s.obs.get(r,'pos_base_x'),s.obs.get(r,'pos_base_y'),s.obs.get(r,'pos_base_rot')])
    def grip(s): return s.obs.get(s.rob(),'pos_gripper')
    def P(s, name): o=s.obs.get_object_from_name(name); return np.array([s.obs.get(o,f) for f in ['x','y','z']])
    def Q(s, name): o=s.obs.get_object_from_name(name); return np.array([s.obs.get(o,f) for f in ['qw','qx','qy','qz']])
    def step(s, db=(0,0,0), dq=None):
        a = np.zeros(11); a[0:3]=np.clip(db,-.1,.1)
        if dq is not None:
            a[3:10]=np.clip(dq,-.1,.1); s.qi = s.qi + 0.25*a[3:10]
        a[10]=s.g
        s.obs,r,t,tr,info = s.env.step(a); s.rew.append(r); return r
    def goto_q(s, qt, maxsteps=400, settle=5):
        for k in range(maxsteps):
            d=(qt-s.qi)/0.25
            if np.max(np.abs(d))<1e-6: break
            s.step(dq=d)
        for k in range(settle): s.step(dq=np.zeros(7))
    def goto_base(s, target, steps=100):
        for k in range(steps):
            d = np.array(target)-s.base()
            if np.max(np.abs(d))<0.002: break
            s.step(db=d)
    def ik_world(s, pw, Rw=RD, base=None, q0=None, tool=None, mount=None):
        b = s.base() if base is None else base
        pa = kin.world_to_arm(pw, b, mount)
        Ra = kin.rotz(-b[2]) @ Rw
        q,err = kin.ik_arm(pa, Ra, s.qi if q0 is None else q0, iters=300, lam=0.05, tool=tool)
        return q, err
    def gripper(s, v, steps=15):
        s.g=v
        for k in range(steps): s.step(dq=np.zeros(7))
    def fk(s, q=None, tool=None, mount=None):
        return kin.fk_world(s.base(), s.q() if q is None else q, tool, mount)
