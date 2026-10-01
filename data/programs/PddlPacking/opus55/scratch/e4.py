from env_client import make_env
import numpy as np
env=make_env(); T=env.observation_space.get_type
f=env.observation_space.type_features[T("robot")]
def rs(o):
    r=o.get_objects(T("robot"))[0]; return np.array([o.get(r,x) for x in f])
def step(d):
    a=np.zeros(11,np.float32); a[:len(d)]=d; o,*_=env.step(a); return o
o,_=env.reset(seed=0); s0=rs(o); o=step((-3,0,0)); print("dx-3",(rs(o)-s0)[:3])
o,_=env.reset(seed=0); s0=rs(o); o=step((0,0,0,3)); print("dq1 3",(rs(o)-s0)[:5].round(3))
o,_=env.reset(seed=0); o=step((0.9,)); print("dx+0.9 ->",rs(o)[:3])  # base at -0.1 overlaps table
