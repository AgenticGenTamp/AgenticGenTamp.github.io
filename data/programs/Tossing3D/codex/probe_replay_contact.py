"""Replay saved contact and inspect/try grasp around first collision."""
import numpy as np
from env_client import make_env

FS=tuple("pos_arm_joint%d"%i for i in range(1,8))
def v(s,n,fs):
 o=s.get_object_from_name(n); return np.array([float(s.get(o,f)) for f in fs])

def replay(grip):
 data=np.load("random_contact_found.npz"); acts=data["actions"]
 env=make_env(); s,_=env.reset(seed=0,options={"object_count":1}); p0=v(s,"cube_0",("x","y","z"))
 for t,a0 in enumerate(acts):
  a=a0.copy()
  if t>=141: a[10]=grip
  qb=v(s,"robot",FS); pb=v(s,"cube_0",("x","y","z"))
  s,*_=env.step(a); pa=v(s,"cube_0",("x","y","z"))
  if t>=139: print("g",grip,"t",t,"base",np.round(v(s,"robot",("pos_base_x","pos_base_y","pos_base_rot")),4),"qb",np.round(qb,4),"qa",np.round(v(s,"robot",FS),4),"d",np.round(pa-p0,6),"step",np.round(pa-pb,6))
 # Hold collision q and closure briefly, then retreat base diagonally.
 qhold=v(s,"robot",FS)
 for k in range(20):
  a=np.zeros(18,np.float32); a[3:10]=np.clip(qhold-v(s,"robot",FS),-.1,.1); a[10]=grip
  if k>=8: a[:2]=[-.06,-.02]
  s,*_=env.step(a)
 print("RESULT grip",grip,"cube delta",np.round(v(s,"cube_0",("x","y","z"))-p0,5),"qhold",np.round(qhold,5))
 env.close()
for g in (0.,1.): replay(g)
