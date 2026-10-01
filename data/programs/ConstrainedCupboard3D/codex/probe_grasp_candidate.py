"""Verify the visually calibrated floor grasp and attachment."""
import numpy as np
from env_client import make_env

e=make_env(); s,_=e.reset(seed=1)
rod=s.get_object_from_name("cuboid_1"); rob=s.get_object_from_name("robot")
def g(o,f): return float(s.get(o,f))
orig=np.array([g(rod,f) for f in ("x","y","z")])
target=np.array([0,0,np.pi,.5,0,-.3,np.pi/2])
for k in range(180):
 a=np.zeros(11,np.float32); a[-1]=1
 if k<15:
  a[0]=np.clip((orig[0]-.55-g(rob,"pos_base_x"))/.87,-.1,.1)
  a[1]=np.clip((orig[1]-g(rob,"pos_base_y"))/.87,-.1,.1)
 elif k<125:
  q=np.array([g(rob,f"pos_arm_joint{i}") for i in range(1,8)])
  er=(target-q+np.pi)%(2*np.pi)-np.pi
  a[3:10]=np.clip(.5*er,-.1,.1)
 elif k<140:
  a[-1]=0
 else:
  a[-1]=0; a[0]=-.04
 s,r,t,tr,i=e.step(a)
 if k%5==0:
  pos=np.array([g(rod,f) for f in ("x","y","z")])
  print(k,"base",round(g(rob,"pos_base_x"),3),round(g(rob,"pos_base_y"),3),"rod",np.round(pos,3),"d",np.round(pos-orig,3),flush=True)
e.close()
