import numpy as np
from env_client import make_env


def val(s, n, f): return float(s.get(s.get_object_from_name(n), f))
def pos(s,n): return np.array([val(s,n,"x"),val(s,n,"y")])
def robot(s): return np.array([val(s,"robot","pos_base_x"),val(s,"robot","pos_base_y")])

e=make_env(); s,info=e.reset(seed=0)
cubes=sorted(n for n in s.get_object_names() if n.startswith("cube_"))
total=0
def drive(target, limit=25):
 global s,total
 for _ in range(limit):
  d=target-robot(s)
  if np.linalg.norm(d)<.06:return
  a=np.zeros(11,np.float32); a[:2]=np.clip(d*2,-.1,.1)
  s,r,t,tr,i=e.step(a);total+=r
  if r != -1: print("reward",r,"total",total,"robot",robot(s),"w",pos(s,"wiper_0"),"c",[pos(s,n).round(2).tolist() for n in cubes])
  if t or tr: print("done",t,tr);return

print("initial",robot(s),pos(s,"wiper_0"),[pos(s,n).round(2).tolist() for n in cubes])
for cycle in range(8):
 w=pos(s,"wiper_0")
 drive(w+np.array([-.15,.60]))
 w=pos(s,"wiper_0")
 drive(w+np.array([-.15,-.50]))
 print("cycle",cycle,"robot",robot(s).round(2),"w",pos(s,"wiper_0").round(2),"c",[pos(s,n).round(2).tolist() for n in cubes],"total",total)
e.close()
