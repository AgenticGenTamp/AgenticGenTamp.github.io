"""Use the wrist contact as a continuous pusher instead of requiring a grasp."""
import math
import numpy as np
from env_client import make_env

def v(s,n,f): return float(s.get(s.get_object_from_name(n),f))
def xy(s,n):
 fs=("pos_base_x","pos_base_y") if n=="robot" else ("x","y")
 return np.array([v(s,n,f) for f in fs])
def q(s): return np.array([v(s,"robot",f"pos_arm_joint{i}") for i in range(1,8)])
def ae(a,b): return (a-b+math.pi)%(2*math.pi)-math.pi

e=make_env();s,_=e.reset(seed=0,options={"object_count":1})
w0=xy(s,"wiper_0"); c0=xy(s,"cube_0")
target=np.array([1.3378,1.0894]);qt=np.array([0,.8,2.37,-2.57,.02,-.8,1.57])
for k in range(45):
 a=np.zeros(11,np.float32);a[:2]=np.clip(.8*(target-xy(s,"robot")),-.1,.1)
 a[2]=np.clip(.8*ae(-1.475,v(s,"robot","pos_base_rot")),-.1,.1)
 a[3:10]=np.clip(.7*(qt-q(s)),-.1,.1);a[10]=1;s,*_=e.step(a)
print("contact",(xy(s,"wiper_0")-w0).round(3),flush=True)
for k in range(45):
 a=np.zeros(11,np.float32);a[1]=-.025;a[3:10]=np.clip(.5*(qt-q(s)),-.1,.1);a[10]=1
 s,r,d,tr,_=e.step(a)
 if k%5==4 or r!=-1 or d:
  print(k+1,"r",r,"base",xy(s,"robot").round(3),"w",xy(s,"wiper_0").round(3),
        "cube",xy(s,"cube_0").round(3),"dw",(xy(s,"wiper_0")-w0).round(3),
        "dc",(xy(s,"cube_0")-c0).round(3),"done",d,tr,flush=True)
 if d or tr:break
e.close()
