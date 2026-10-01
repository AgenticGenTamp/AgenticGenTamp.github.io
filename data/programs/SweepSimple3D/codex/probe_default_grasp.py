import numpy as np
from env_client import make_env
def g(s,n,f): return float(s.get(s.get_object_from_name(n),f))
def pos(s,n):
 fs=('pos_base_x','pos_base_y','pos_base_rot') if n=='robot' else ('x','y','z')
 return np.array([g(s,n,f) for f in fs])
for grip in (0.,1.):
 e=make_env();s,_=e.reset(seed=0,options={'object_count':1});w0=pos(s,'wiper_0')
 for t in range(8):
  a=np.zeros(11,np.float32);a[10]=grip;s,*_=e.step(a)
 for t in range(16):
  a=np.zeros(11,np.float32);a[1]=-.04;a[10]=grip;s,r,d,tr,_=e.step(a)
  print('g',grip,t,'b',pos(s,'robot').round(2),'w',pos(s,'wiper_0').round(2),'dw',(pos(s,'wiper_0')-w0).round(2),'gp',g(s,'robot','pos_gripper'))
 e.close()
