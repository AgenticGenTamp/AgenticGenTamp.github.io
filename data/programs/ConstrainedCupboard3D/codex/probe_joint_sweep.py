import sys
import numpy as np
from env_client import make_env

joint=int(sys.argv[1]) if len(sys.argv)>1 else 2
sign=float(sys.argv[2]) if len(sys.argv)>2 else 1.0
e=make_env(); s,_=e.reset(seed=1); n='cuboid_1'; o=s.get_object_from_name(n); rob=s.get_object_from_name('robot')
def v(obj,f): return float(s.get(obj,f))
orig=np.array([v(o,f) for f in ('x','y','z')])
# move base to x=.1, y=.12
for k in range(3):
 a=np.zeros(11,np.float32); a[1]=.08; e.step(a)
for k in range(100):
 a=np.zeros(11,np.float32); a[2+joint]=.1*sign; a[10]=0
 s,r,t,tr,inf=e.step(a)
 p=np.array([v(o,f) for f in ('x','y','z')])
 q=v(rob,'pos_arm_joint'+str(joint))
 if k%10==0: print(k,'q',round(q,3),'move',round(float(np.linalg.norm(p-orig)),4),flush=True)
 if np.linalg.norm(p-orig)>.003: print('HIT',k,q,p,flush=True); break
e.close()
