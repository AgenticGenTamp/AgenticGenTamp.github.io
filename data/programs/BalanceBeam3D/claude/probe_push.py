import numpy as np
from env_client import make_env
env=make_env()
obs,info=env.reset(seed=0)
o=np.asarray(obs,float)
print("start base",np.round(o[16:19],3),"lb",np.round(o[0:3],3))
a=np.zeros(11,dtype=np.float32)
prev=o.copy()
for t in range(120):
    a[:]=0
    a[0]=0.1   # +x base
    if t>40: a[1]=-0.05
    obs,r,term,trunc,info=env.step(a)
    o=np.asarray(obs,float)
    if t%20==0 or r!=-1.0:
        print(t,"r",r,"base",np.round(o[16:19],3),"lb",np.round(o[0:3],3),"sb1",np.round(o[54:57],3),"ssq",np.round(o[41:45],3))
    if term or trunc: print("END",t,term,trunc); break
