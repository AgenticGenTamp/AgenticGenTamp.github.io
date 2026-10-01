"""Compensate elbow forward motion to descend vertically around a cube."""
from env_client import make_env
import numpy as np

for factor in (0.,.1,.2,.3):
 for q7 in (0.,np.pi/2):
  low_q4=-1.4;env=make_env();s,_=env.reset(seed=0,options={"object_count":4});names=sorted(n for n in s.get_object_names() if n.startswith("cube"));pall={n:np.array([s.get(s.get_object_from_name(n),f) for f in ("x","y","z")]) for n in names};p0=pall["cube3"];high=np.array([0.,1.34,np.pi,-1.,0.,1.,q7]);low=high.copy();low[3]=low_q4;bx0=.65;bx1=bx0+factor*(-1.-low_q4)
  def go(bg,qg,g,n):
   global s
   for _ in range(n):
    r=s.get_object_from_name("robot");b=np.array([s.get(r,f) for f in ("pos_base_x","pos_base_y","pos_base_rot")]);q=np.array([s.get(r,f"pos_arm_joint{i}") for i in range(1,8)]);a=np.zeros(11,np.float32);e=bg-b;e[2]=(e[2]+np.pi)%(2*np.pi)-np.pi;a[:3]=np.clip(e,-.1,.1);a[3:10]=np.clip(qg-q,-.1,.1);a[10]=g;s,_,_,_,_=env.step(a)
  bh=np.array([bx0,p0[1],np.pi]);bl=np.array([bx1,p0[1],np.pi]);go(np.array([1.,p0[1],np.pi]),high,1.,70);go(bh,high,1.,20);go(bl,low,1.,25);go(bl,low,0.,15);go(bh,high,0.,30)
  final={n:np.array([s.get(s.get_object_from_name(n),f) for f in ("x","y","z")]) for n in names};best=max(names,key=lambda n:np.linalg.norm(final[n]-pall[n]));print(factor,round(q7,2),round(bx1,3),best,np.round(final[best]-pall[best],3).tolist());env.close()
