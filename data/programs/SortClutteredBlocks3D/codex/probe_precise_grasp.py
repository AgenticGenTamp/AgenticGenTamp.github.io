"""Stop at first open-jaw contact, close, then reverse the arm trajectory."""
from env_client import make_env
import numpy as np

for q7 in (0.,np.pi/2):
 env=make_env();s,_=env.reset(seed=0,options={"object_count":4});names=sorted(n for n in s.get_object_names() if n.startswith("cube"));p0={n:np.array([s.get(s.get_object_from_name(n),f) for f in ("x","y","z")]) for n in names};cy=p0["cube3"][1];low=np.array([0.,1.34,np.pi,-1.4,0.,1.,q7])
 def act(bg,qg,g,limit=.1):
  global s
  r=s.get_object_from_name("robot");b=np.array([s.get(r,f) for f in ("pos_base_x","pos_base_y","pos_base_rot")]);q=np.array([s.get(r,f"pos_arm_joint{i}") for i in range(1,8)]);a=np.zeros(11,np.float32);e=bg-b;e[2]=(e[2]+np.pi)%(2*np.pi)-np.pi;a[:3]=np.clip(e,-limit,limit);a[3:10]=np.clip(qg-q,-.1,.1);a[10]=g;s,_,_,_,_=env.step(a);return b
 for _ in range(80):act(np.array([1.,cy,np.pi]),low,1.)
 hit=None
 for k in range(100):
  b=act(np.array([.6,cy,np.pi]),low,1.,.006)
  moved=max(np.linalg.norm(np.array([s.get(s.get_object_from_name(n),f) for f in ("x","y","z")])-p0[n]) for n in names)
  if moved>.001:hit=b.copy();break
 if hit is not None:
  hold=hit+np.array([.004,0,0]);
  for _ in range(8):act(hold,low,1.,.01)
  for _ in range(20):act(hold,low,0.)
  high=low.copy();high[3]=-1.;bh=hold+np.array([.12,0,0])
  for _ in range(35):act(bh,high,0.)
 p={n:np.array([s.get(s.get_object_from_name(n),f) for f in ("x","y","z")]) for n in names};print(round(q7,2),"hit",None if hit is None else np.round(hit,3).tolist(),"d3",np.round(p["cube3"]-p0["cube3"],3).tolist(),"peakz",round(float(p["cube3"][2]),3));env.close()
