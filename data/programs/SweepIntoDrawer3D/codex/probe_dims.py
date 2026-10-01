import numpy as np
from env_client import make_env

for j in range(11):
  for sgn in (1,-1) if j<10 else (1,):
    e=make_env(); o,_=e.reset(seed=0); start=o.copy(); rew=[]
    a=np.zeros(11,np.float32); a[j]=0.1*sgn if j<10 else float(sgn)
    for k in range(10): o,r,t,tr,inf=e.step(a); rew.append(r)
    print(j,sgn,'base',np.round(o[125:128]-start[125:128],3),'joints',np.round(o[128:136]-start[128:136],3),'wxyz',np.round(o[147:150]-start[147:150],3),'draw',np.round(o[103:109],3),'cubes',np.round((o.reshape(-1)[:80].reshape(5,16)[:,:3]-start[:80].reshape(5,16)[:,:3]),3),'r',rew[-1])
    e.close()
