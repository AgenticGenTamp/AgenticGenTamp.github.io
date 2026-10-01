import sys,numpy as np
from env_client import make_env
mode=sys.argv[1]
env=make_env(); obs,info=env.reset(seed=91)
def rb(o):
    r=o.get_object_from_name('robot'); return np.array([float(o.get(r,'pos_base_x')),float(o.get(r,'pos_base_y'))])
rng=np.random.default_rng(0)
best=0
for i in range(400):
    a=np.zeros(11,dtype=np.float32)
    if mode=='dir': a[:2]=[0.1,0.1]
    elif mode=='rot': a[2]=0.1
    elif mode=='rand': a=env.action_space.sample()
    elif mode=='arm': a[3:10]=0.1*np.sin(i/5.0)
    elif mode=='wait': pass
    obs,r,term,trunc,info=env.step(a)
    b=rb(obs); best=max(best,float(np.linalg.norm(b)))
    if term: print(mode,"TERM",i); break
print(mode,"final",np.round(rb(obs),3),"maxnorm",round(best,3))
