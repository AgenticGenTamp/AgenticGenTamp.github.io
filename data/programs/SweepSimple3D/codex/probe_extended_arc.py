"""Extend the known aligned arm posture from a chassis-safe base pose."""
import math
import numpy as np
from env_client import make_env

def v(s,n,f): return float(s.get(s.get_object_from_name(n),f))
def p(s,n): return np.array([v(s,n,f) for f in "xyz"])
def b(s): return np.array([v(s,"robot","pos_base_x"),v(s,"robot","pos_base_y")])
def q(s): return np.array([v(s,"robot",f"pos_arm_joint{i}") for i in range(1,8)])
def ae(a,x): return (a-x+math.pi)%(2*math.pi)-math.pi

e=make_env()
for q6 in (-.8,):
 s,_=e.reset(seed=0,options={"object_count":1}); w0=p(s,"wiper_0")
 # 35.5 cm from handle: beyond the requested chassis-safe threshold.
 bt=np.array([1.3378,1.171]); yaw=-1.475
 qt=np.array([-2.7,.6,2.37,-2.57,.02,q6,1.57])
 for k in range(45):
  a=np.zeros(11,np.float32);a[:2]=np.clip(.7*(bt-b(s)),-.1,.1)
  a[2]=np.clip(.7*ae(yaw,v(s,"robot","pos_base_rot")),-.1,.1)
  a[3:10]=np.clip(.7*(qt-q(s)),-.1,.1);a[10]=1
  s,*_=e.step(a)
 for k in range(58):
  qb=q(s).copy();wb=p(s,"wiper_0").copy();a=np.zeros(11,np.float32)
  a[3]=.1;a[10]=0;s,*_=e.step(a);d=p(s,"wiper_0")-wb
  if np.linalg.norm(d)>.001:
   print("CONTACT q6",q6,"step",k,"base",b(s).round(4).tolist(),
    "radius",round(float(np.linalg.norm(b(s)-p(s,"wiper_0")[:2])),4),
    "yaw",round(v(s,"robot","pos_base_rot"),4),"q_before",qb.round(4).tolist(),
    "q_after",q(s).round(4).tolist(),"w_delta",d.round(4).tolist(),flush=True)
 print("DONE",q6,"base",b(s).round(3),"q",q(s).round(3),"wd",(p(s,"wiper_0")-w0).round(4),flush=True)
e.close()
