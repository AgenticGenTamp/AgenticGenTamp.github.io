from env_client import make_env
import numpy as np
env = make_env()
R=None
obs,_=env.reset(seed=0)
rt=[t for t in env.observation_space.type_features if t.name=='robot'][0]
r=obs.get_objects(rt)[0]
feats=env.observation_space.type_features[rt]
def rs(o): return [round(float(o.get(r,f)),3) for f in feats[:12]]
def step(a):
    a=np.array(a,dtype=np.float32); o,rew,term,trunc,info=env.step(a); return o,rew,term,trunc
print(rs(obs))
o,*x=step([0.2,0,0,0,0,0,0,0,0,0,0]); print('dx',rs(o),x)
o,*x=step([0,0,0.2,0,0,0,0,0,0,0,0]); print('rot',rs(o))
o,*x=step([0.2,0,0,0,0,0,0,0,0,0,0]); print('dx after rot',rs(o))
o,*x=step([0,0.2,0,0,0,0,0,0,0,0,0]); print('dy after rot',rs(o))
o,*x=step([0,0,0,0,0,0,0,0,0,0,-1]); print('close',rs(o))
o,*x=step([0,0,0,0,0,0,0,0,0,0,1]); print('open',rs(o))
o,*x=step([0,0,0,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0]); print('joints',rs(o))
for i in range(20):
    o,*x=step([0,0,0,0,-0.2,0,0,0,0,0,0])
print('j2 down x20',rs(o))
for i in range(20):
    o,*x=step([0,0,0,0,0,0,0,0,0.2,0,0])
print('j6 x20',rs(o))
env.close()
