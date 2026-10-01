from env_client import make_env
import numpy as np
env=make_env()
obs,_=env.reset(seed=0)
def flat(o):
    d={}
    for t in env.observation_space.types:
        for ob in o.get_objects(t):
            for f in env.observation_space.type_features[t]: d[(ob.name,f)]=o.get(ob,f)
    return d
prev=flat(obs)
for v in [-1,-0.6,-0.51,0.6,1.0,-1.0]:
    a=np.zeros(11,dtype=np.float32); a[10]=v
    obs,rw,te,tr,inf=env.step(a); cur=flat(obs)
    print(v,{k:(prev[k],cur[k]) for k in cur if cur[k]!=prev.get(k)}, inf)
    prev=cur
print(type(obs), [m for m in dir(obs) if not m.startswith('__')])
env.close()
