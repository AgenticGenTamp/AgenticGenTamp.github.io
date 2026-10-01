"""Test finger-plane rotations at the staged top-handle contact."""
import math
import numpy as np
from env_client import make_env

def v(s,n,f): return float(s.get(s.get_object_from_name(n),f))
def xy(s,n):
 fs=("pos_base_x","pos_base_y") if n=="robot" else ("x","y")
 return np.array([v(s,n,f) for f in fs])
def q(s): return np.array([v(s,"robot",f"pos_arm_joint{i}") for i in range(1,8)])
def ae(a,b): return (a-b+math.pi)%(2*math.pi)-math.pi

e=make_env()
for q7 in (0,.8,1.57,2.35,3.14):
 s,_=e.reset(seed=0,options={"object_count":1});w0=xy(s,"wiper_0")
 final=np.array([1.3378,1.0894]);qt=np.array([0,.8,2.37,-2.57,.02,-.8,q7])
 # Stop north of contact open, close, then ingress.
 for _ in range(42):
  a=np.zeros(11,np.float32);a[:2]=np.clip(.8*(final+[0,.09]-xy(s,"robot")),-.1,.1)
  a[2]=np.clip(.8*ae(-1.475,v(s,"robot","pos_base_rot")),-.1,.1)
  a[3:10]=np.clip(.7*(qt-q(s)),-.1,.1);a[10]=1;s,*_=e.step(a)
 for _ in range(7):
  a=np.zeros(11,np.float32);a[3:10]=np.clip(.5*(qt-q(s)),-.1,.1);a[10]=0;s,*_=e.step(a)
 for _ in range(10):
  a=np.zeros(11,np.float32);a[:2]=np.clip(.5*(final-xy(s,"robot")),-.04,.04)
  a[3:10]=np.clip(.5*(qt-q(s)),-.1,.1);a[10]=0;s,*_=e.step(a)
 pre=xy(s,"wiper_0").copy()
 for _ in range(8):
  a=np.zeros(11,np.float32);a[1]=.06;a[3:10]=np.clip(.5*(qt-q(s)),-.1,.1);a[10]=0;s,*_=e.step(a)
 print(q7,"contact",(pre-w0).round(3),"tug",(xy(s,"wiper_0")-pre).round(3),flush=True)
e.close()
