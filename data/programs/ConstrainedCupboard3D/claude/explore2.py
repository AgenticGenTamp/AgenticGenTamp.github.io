import numpy as np
from env_client import make_env
env = make_env()
obs, info = env.reset(seed=0)
def rob(o):
    r=o.get_object_from_name('robot')
    return np.array(o.data[r])
def cub(o,i):
    c=o.get_object_from_name('cuboid_%d'%i)
    return np.array(o.data[c])
r0=rob(obs)
feats=obs.type_features[obs.get_object_from_name('robot').type]
for k in range(11):
    a=np.zeros(11,dtype=np.float32); a[k]=0.1 if k<10 else 1.0
    o,rew,t,tr,inf=env.step(a)
    r1=rob(o)
    d=r1-r0
    print(k, "rew",round(rew,4), {feats[i]:round(float(d[i]),4) for i in range(11) if abs(d[i])>1e-4})
    r0=r1
env.close()
