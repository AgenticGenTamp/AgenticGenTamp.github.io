"""Focused arm-only grasp validation around the rendered pickup alignment."""
import math
import numpy as np
from env_client import make_env

def v(s,n,f): return float(s.get(s.get_object_from_name(n),f))
def xyz(s,n): return np.array([v(s,n,f) for f in "xyz"])
def q(s): return np.array([v(s,"robot",f"pos_arm_joint{i}") for i in range(1,8)])
def base(s): return np.array([v(s,"robot","pos_base_x"),v(s,"robot","pos_base_y")])
def ae(a,b): return (a-b+math.pi)%(2*math.pi)-math.pi

e=make_env(); rng=np.random.default_rng(771); yaw=-1.475
R=np.array([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]])
for trial in range(28):
 s,_=e.reset(seed=0,options={"object_count":1}); w0=xyz(s,"wiper_0")
 fwd,lat=rng.uniform(-.18,.18,2); q2=rng.uniform(.2,1.45); q4=rng.uniform(-2.85,-1.8)
 qt=np.array([rng.uniform(-.35,.35),q2,2.37,q4,rng.uniform(-.3,.3),rng.uniform(-1.2,-.4),1.57])
 bt=w0[:2]-R@np.array([.400+fwd,.360+lat])
 for k in range(42):
  a=np.zeros(11,np.float32);a[:2]=np.clip(.7*(bt-base(s)),-.1,.1)
  a[2]=np.clip(.8*ae(yaw,v(s,"robot","pos_base_rot")),-.1,.1)
  a[3:10]=np.clip(.7*(qt-q(s)),-.1,.1);a[10]=1.;s,*_=e.step(a)
 # settle open, close with no base or arm motion
 for k in range(3):
  a=np.zeros(11,np.float32);a[10]=1.;s,*_=e.step(a)
 pre=xyz(s,"wiper_0").copy()
 for k in range(7):
  a=np.zeros(11,np.float32);a[10]=0.;s,*_=e.step(a)
 grip=v(s,"robot","pos_gripper"); closed=xyz(s,"wiper_0").copy()
 # Arm-only shoulder lift then reverse. Base entries remain exactly zero.
 qa=q(s).copy(); hi=qa.copy();hi[1]+=0.38
 for k in range(7):
  a=np.zeros(11,np.float32);a[3:10]=np.clip(.7*(hi-q(s)),-.1,.1);s,*_=e.step(a)
 top=xyz(s,"wiper_0").copy()
 for k in range(7):
  a=np.zeros(11,np.float32);a[3:10]=np.clip(.7*(qa-q(s)),-.1,.1);s,*_=e.step(a)
 end=xyz(s,"wiper_0").copy()
 print(trial,"fl",np.round([fwd,lat],3).tolist(),"q",np.round(qt,3).tolist(),
       "g",round(grip,4),"place",np.round(pre-w0,4).tolist(),
       "close",np.round(closed-pre,4).tolist(),"up",np.round(top-closed,4).tolist(),
       "back",np.round(end-top,4).tolist(),flush=True)
 if (np.linalg.norm(top-closed)>.025 and np.linalg.norm(end-top)>.025) or top[2]-closed[2]>.035:
  print("STRONG",trial,flush=True);break
e.close()
