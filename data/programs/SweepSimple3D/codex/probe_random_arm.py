import numpy as np
from env_client import make_env

def v(s,n,f): return float(s.get(s.get_object_from_name(n),f))
def q(s): return np.array([v(s,'robot',f'pos_arm_joint{i}') for i in range(1,8)])
def p(s,n): return np.array([v(s,n,f) for f in 'xyz'])

rng=np.random.default_rng(2);e=make_env();s,_=e.reset(seed=0)
for _ in range(4):
 a=np.zeros(11,np.float32);a[1]=-.1;a[10]=1;s,_,_,_,_=e.step(a)
wlast=p(s,'wiper_0'); contact=False
lo=np.array([-3.1,-2.1,-3.1,-2.5,-3.1,-1.9,-3.1]);hi=-lo
for trial in range(25):
 target=rng.uniform(lo,hi); target[3]=rng.uniform(-2.4,1.0)
 for i in range(28):
  a=np.zeros(11,np.float32);a[3:10]=np.clip((target-q(s))*.7,-.1,.1);a[10]=1
  s,r,d,tr,_=e.step(a);w=p(s,'wiper_0')
  if np.linalg.norm(w-wlast)>.003:
   print('CONTACT',trial,i,'q',q(s).round(3).tolist(),'target',target.round(3).tolist(),'w',w.round(3).tolist(),'delta',(w-wlast).round(3).tolist(),flush=True);contact=True
  wlast=w
 print('trial',trial,'q',q(s).round(2).tolist(),'contact',contact,flush=True)
 if contact:break
e.close()
