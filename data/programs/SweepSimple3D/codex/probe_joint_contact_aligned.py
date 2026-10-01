"""At the visually aligned base pose, sweep arm joints to find handle contact."""

import math
import numpy as np
from env_client import make_env


def v(s, n, f): return float(s.get(s.get_object_from_name(n), f))
def w(s): return np.array([v(s, "wiper_0", f) for f in "xyz"])
def base(s): return np.array([v(s, "robot", "pos_base_x"), v(s, "robot", "pos_base_y")])
def q(s): return np.array([v(s, "robot", "pos_arm_joint%d" % i) for i in range(1, 8)])


def run(env, dim, sign):
    s, _ = env.reset(seed=0, options={"object_count": 1}); w0 = w(s); q0 = q(s)
    target = w0[:2] + np.array([-.36, .40])
    for _ in range(15):
        a = np.zeros(11, np.float32); a[:2] = np.clip(target-base(s), -.1, .1)
        er = (-math.pi/2-v(s,"robot","pos_base_rot")+math.pi)%(2*math.pi)-math.pi
        a[2] = np.clip(er, -.1, .1); a[4] = np.clip(q0[1]+.5-q(s)[1],-.1,.1)
        s,_,_,_,_=env.step(a)
    for k in range(45):
        a=np.zeros(11,np.float32);a[4]=np.clip(q0[1]+.5-q(s)[1],-.1,.1)
        a[3+dim]=sign*.1
        s,_,_,_,_=env.step(a); delta=w(s)-w0
        if np.linalg.norm(delta)>.002:
            return k+1, q(s), delta
    return None, q(s), w(s)-w0


if __name__ == '__main__':
    e=make_env()
    for dim in (3,4,5,6): # arm joint4 through joint7, zero-based
        for sign in (-1,1):
            k,qq,d=run(e,dim,sign)
            print('joint',dim+1,'sign',sign,'first',k,'q',np.round(qq,3),'wd',np.round(d,4),flush=True)
    e.close()
