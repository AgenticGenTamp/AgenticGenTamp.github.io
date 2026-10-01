"""Sweep shoulder yaw at useful base poses/heights to locate wiper with arm."""
import math
import numpy as np
from env_client import make_env

def v(s,n,f): return float(s.get(s.get_object_from_name(n),f))
def xy(s,n):
    fs=("pos_base_x","pos_base_y") if n=="robot" else ("x","y")
    return np.array([v(s,n,f) for f in fs])

e=make_env()
for q2off in (-.5,0,.5,1.0,1.5):
 s,_=e.reset(seed=0,options={"object_count":1}); w0=xy(s,"wiper_0"); q20=v(s,"robot","pos_arm_joint2")
 # Pose used by policy, where rendered fingers are close to the shaft.
 target=w0+np.array([-.26,.47])
 for k in range(28):
  a=np.zeros(11,np.float32);a[:2]=np.clip(target-xy(s,"robot"),-.1,.1)
  yaw=v(s,"robot","pos_base_rot");a[2]=np.clip((-1.80-yaw+math.pi)%(2*math.pi)-math.pi,-.1,.1)
  a[3]=np.clip(-1.0-v(s,"robot","pos_arm_joint1"),-.1,.1)
  a[4]=np.clip(q20+q2off-v(s,"robot","pos_arm_joint2"),-.1,.1)
  s,*_=e.step(a)
 before=xy(s,"wiper_0").copy(); first=None
 for k in range(55):
  a=np.zeros(11,np.float32);a[3]=.1;s,*_=e.step(a)
  d=xy(s,"wiper_0")-before
  if np.linalg.norm(d)>.002 and first is None:first=(k,round(v(s,"robot","pos_arm_joint1"),3),d.round(3).tolist())
 print("q2off",q2off,"actual",round(v(s,"robot","pos_arm_joint2"),2),"first",first,
       "wdelta",(xy(s,"wiper_0")-w0).round(3).tolist(),flush=True)
e.close()
