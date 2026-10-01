import numpy as np
from env_client import make_env
np.set_printoptions(precision=4,suppress=True)
e=make_env()
s,i=e.reset(seed=0)
print('initial',i,'robot',s[125:147],'island',s[96:109],'wiper',s[147:],'cubes',s[:80].reshape(5,16)[:,:3],flush=True)
for j in range(11):
 s,i=e.reset(seed=0)
 a=np.zeros(11,dtype=np.float32); a[j]=.1 if j<10 else 1
 t,r,d,tr,i=e.step(a)
 print('action',j,'delta',t[125:136]-s[125:136],'reward',r,'info',i,'cube_delta',np.max(abs(t[:80]-s[:80])),flush=True)
for g in [0,1]:
 s,i=e.reset(seed=0)
 a=np.zeros(11,dtype=np.float32); a[10]=g
 for k in range(10): t,r,d,tr,i=e.step(a)
 print('grip',g,'robot',t[125:136],'reward',r,'wiper',t[147:154],flush=True)
e.close()
