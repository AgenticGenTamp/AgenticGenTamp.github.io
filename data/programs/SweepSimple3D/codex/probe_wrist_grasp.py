"""Focused elbow/wrist grid at rendered pickup pose; close and tug each candidate."""
import math
import numpy as np
from env_client import make_env

def v(s,n,f): return float(s.get(s.get_object_from_name(n),f))
def xy(s,n):
 fs=("pos_base_x","pos_base_y") if n=="robot" else ("x","y")
 return np.array([v(s,n,f) for f in fs])
def qs(s): return np.array([v(s,"robot","pos_arm_joint%d"%i) for i in range(1,8)])

e=make_env()
for q4 in (-2.9,-2.5,-2.1,-1.7,-1.3):
 for q6 in (-1.5,-.9,-.3,.3):
  s,_=e.reset(seed=0,options={"object_count":1});w0=xy(s,"wiper_0"); q0=qs(s)
  target=w0+np.array([-.26,.47]); qt=q0.copy();qt[1]+=.5;qt[3]=q4;qt[5]=q6
  for k in range(30):
   a=np.zeros(11,np.float32);a[:2]=np.clip(target-xy(s,"robot"),-.1,.1)
   a[2]=np.clip((-1.8-v(s,"robot","pos_base_rot")+math.pi)%(2*math.pi)-math.pi,-.1,.1)
   a[3:10]=np.clip(.7*(qt-qs(s)),-.1,.1);a[10]=1.;s,*_=e.step(a)
  contact=(xy(s,"wiper_0")-w0).copy(); pre=xy(s,"wiper_0").copy()
  for k in range(8):
   a=np.zeros(11,np.float32);a[10]=0.;s,*_=e.step(a)
  for k in range(5):
   a=np.zeros(11,np.float32);a[:2]=[0,.08];a[10]=0.;s,*_=e.step(a)
  follow=xy(s,"wiper_0")-pre
  print("q4/q6",q4,q6,"actual",qs(s).round(2).tolist(),"contact",contact.round(3).tolist(),
        "tug",follow.round(3).tolist(),flush=True)
e.close()
