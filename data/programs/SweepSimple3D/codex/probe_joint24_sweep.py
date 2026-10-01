"""Map shoulder/elbow sweeps near the visually aligned wiper pose."""
import math
import numpy as np
from env_client import make_env

def v(s,n,f): return float(s.get(s.get_object_from_name(n),f))
def xy(s,n):
 fs=("pos_base_x","pos_base_y") if n=="robot" else ("x","y")
 return np.array([v(s,n,f) for f in fs])

e=make_env()
for q1off in (-.7,0,.7):
 for dim in (4,6):
  s,_=e.reset(seed=0,options={"object_count":1});w0=xy(s,"wiper_0"); target=w0+[-.26,.47]
  q10=v(s,"robot","pos_arm_joint1"); qstart=v(s,"robot","pos_arm_joint%d"%(dim-2))
  # Put selected joint roughly 0.8 radians below nominal, then scan upward.
  for k in range(35):
   a=np.zeros(11,np.float32);a[:2]=np.clip(target-xy(s,"robot"),-.1,.1)
   a[2]=np.clip((-1.8-v(s,"robot","pos_base_rot")+math.pi)%(2*math.pi)-math.pi,-.1,.1)
   a[3]=np.clip(q10+q1off-v(s,"robot","pos_arm_joint1"),-.1,.1)
   feat="pos_arm_joint%d"%(dim-2)
   a[dim]=np.clip(qstart-.8-v(s,"robot",feat),-.1,.1);s,*_=e.step(a)
  before=xy(s,"wiper_0").copy(); first=None
  for k in range(70):
   a=np.zeros(11,np.float32);a[dim]=.1;s,*_=e.step(a);d=xy(s,"wiper_0")-before
   if np.linalg.norm(d)>.002 and first is None:first=(k,round(v(s,"robot",feat),2),d.round(3).tolist())
  print("q1off",q1off,"joint",dim-2,"first",first,"wd",(xy(s,"wiper_0")-w0).round(3).tolist(),flush=True)
e.close()
