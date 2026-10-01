import numpy as np
from env_client import make_env
env=make_env(); obs,info=env.reset(seed=2)
def base(o):
    r=o.get_object_from_name('robot'); return np.array([float(o.get(r,f)) for f in ['pos_base_x','pos_base_y','pos_base_rot']])
def chairs(o):
    out={}
    for n in sorted(o.get_object_names()):
        if n=='robot': continue
        c=o.get_object_from_name(n); out[n]=[round(float(o.get(c,f)),3) for f in ['x','y','z']]
    return out
tgt=(2.84,2.40)
for i in range(400):
    b=base(obs); d=np.array(tgt[:2])-b[:2]
    a=np.zeros(11,dtype=np.float32); a[0]=np.clip(d[0],-0.1,0.1); a[1]=np.clip(d[1],-0.1,0.1)
    obs,rew,term,trunc,info=env.step(a)
    if i%20==0 or term: print(i, round(rew,4), term, [round(v,3) for v in base(obs)], chairs(obs))
    if term or trunc: break
