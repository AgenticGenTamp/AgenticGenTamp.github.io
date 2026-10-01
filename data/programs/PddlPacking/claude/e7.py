import numpy as np
from env_client import make_env
RF=["base_x","base_y","base_rot"]
def rob(s):
    r=s.get_object_from_name("robot"); return np.array([s.get(r,f) for f in RF])
env=make_env()
for idx,sgn in [(0,1),(0,-1),(1,1),(1,-1)]:
    obs,info=env.reset(seed=0)
    prev=rob(obs)
    for k in range(40):
        a=np.zeros(11); a[idx]=0.05*sgn
        obs,_,_,_,_=env.step(a)
        c=rob(obs)
        if np.allclose(c,prev): break
        prev=c
    print("idx",idx,"sgn",sgn,"final",np.round(prev,3),"steps",k)
env.close()
