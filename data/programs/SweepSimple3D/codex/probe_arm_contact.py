import sys
import numpy as np
from env_client import make_env


def g(s,n,f): return float(s.get(s.get_object_from_name(n),f))
def joints(s): return np.array([g(s,"robot",f"pos_arm_joint{i}") for i in range(1,8)])
def xyz(s,n): return np.array([g(s,n,f) for f in "xyz"])

lo = int(sys.argv[1]) if len(sys.argv) > 1 else 3
for dim in range(lo,10):
 for sign in (-1,1):
  e=make_env();s,i=e.reset(seed=0);w0=xyz(s,"wiper_0"); q0=joints(s)
  moved=0
  for k in range(35):
   a=np.zeros(11,np.float32);a[dim]=sign*.1
   s,r,t,tr,i=e.step(a)
   wd=np.linalg.norm(xyz(s,"wiper_0")-w0)
   if wd>.002 and moved==0: moved=k+1
  print("dim",dim,"sgn",sign,"qdelta",np.round(joints(s)-q0,2),"wdelta",np.round(xyz(s,"wiper_0")-w0,3),"first",moved)
  e.close()

# Coupled shoulder/elbow patterns.
for ds,de in [(-1,-1),(-1,1),(1,-1),(1,1)]:
 e=make_env();s,i=e.reset(seed=0);w0=xyz(s,"wiper_0");q0=joints(s)
 for k in range(45):
  a=np.zeros(11,np.float32);a[4]=ds*.1;a[6]=de*.1
  s,r,t,tr,i=e.step(a)
 print("combo",ds,de,"qdelta",np.round(joints(s)-q0,2),"wdelta",np.round(xyz(s,"wiper_0")-w0,3))
 e.close()
