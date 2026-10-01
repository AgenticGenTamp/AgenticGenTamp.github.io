import numpy as np
from env_client import make_env

def v(s,n,f): return float(s.get(s.get_object_from_name(n),f))
def p(s,n): return np.array([v(s,n,f) for f in ('x','y','z')])
e=make_env();s,_=e.reset(seed=0); names=['wiper_0']+sorted(n for n in s.get_object_names() if n.startswith('cube_')); p0={n:p(s,n) for n in names}
for i in range(220):
 a=np.zeros(11,np.float32);a[1]=-.01
 s,r,d,tr,_=e.step(a)
 if i%20==19 or r != -1 or d:
  print(i+1,r,'robot',round(v(s,'robot','pos_base_y'),3),{n:np.round(p(s,n)-p0[n],3).tolist() for n in names if np.linalg.norm(p(s,n)-p0[n])>.002},d)
 if d or tr:break
e.close()
