from env_client import make_env
import numpy as np

env=make_env(); obj=np.array([0,1,2,16,17,18,32,33,34])
for axis in [4,6]:
  for first in [1,-1]:
    s0,_=env.reset(seed=0); s=s0
    print('trial',axis,first,'base',s[93:96],'q',s[96:103],flush=True)
    for phase,(sgn,n) in enumerate([(first,35),(-first,70)]):
      a=np.zeros(11,np.float32); a[axis]=sgn*.1
      for k in range(n):
        old=s; s,r,t,tr,_=env.step(a)
        dd=s[obj]-old[obj]; d=s[obj]-s0[obj]
        if np.max(np.abs(dd))>.003 or r!=-1:
          print('CONTACT',phase,k,'r',r,'q',np.round(s[96:103],2),'dd',np.round(dd.reshape(3,3),3),'tot',np.round(d.reshape(3,3),3),flush=True)
    print('end q',s[96:103],flush=True)
env.close()
