import numpy as np
from env_client import make_env

e=make_env(); s,_=e.reset(seed=1); n='cuboid_1'; o=s.get_object_from_name(n); rob=s.get_object_from_name('robot')
def xy(obj,fs): return np.array([s.get(obj,f) for f in fs],float)
orig=xy(o,('x','y','z'))
def go(vx,vy,steps,k):
 global s
 for j in range(steps):
  a=np.zeros(11,np.float32); a[:2]=[vx,vy]; a[10]=0
  s,r,t,tr,inf=e.step(a)
  p=xy(o,('x','y','z')); b=xy(rob,('pos_base_x','pos_base_y','pos_base_rot'))
  if np.linalg.norm(p-orig)>.002: print('HIT',k,j,'base',b,'obj',p); return True
 return False
# Align to rod's y, then scan x from 0 to 1.5 and back.
print('orig',orig)
if go(0,.1,3,'align'): raise SystemExit
if go(.1,0,18,'forward'): raise SystemExit
if go(-.1,0,22,'back'): raise SystemExit
print('NO HIT base',xy(rob,('pos_base_x','pos_base_y','pos_base_rot')))
e.close()
