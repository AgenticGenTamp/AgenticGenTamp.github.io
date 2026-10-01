import sys, numpy as np
from env_client import make_env
def rob(obs):
    r=obs.get_object_from_name("robot"); f=obs.type_features[r.type]; d=obs.data[r]
    return {n:float(d[f.index(n)]) for n in f}
env=make_env()
for dim,sg,nm in [(0,1,'+x'),(0,-1,'-x'),(1,1,'+y'),(1,-1,'-y')]:
    obs,_=env.reset(seed=0); a=np.zeros(11,np.float32); a[dim]=0.1*sg
    key='pos_base_'+('x' if dim==0 else 'y'); prev=rob(obs)[key]; stop=None
    for i in range(120):
        obs,*_=env.step(a); q=rob(obs)[key]
        if stop is None and abs(q-prev)<1e-3 and i>2: stop=(i+1,round(q,3))
        prev=q
    s=rob(obs)
    print("%s stop=%s final x=%.3f y=%.3f rot=%.3f"%(nm,stop,s['pos_base_x'],s['pos_base_y'],s['pos_base_rot']))
env.close()
