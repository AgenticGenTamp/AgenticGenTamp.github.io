"""Focused close-and-lift grid around visually aligned short arm poses."""
from env_client import make_env
import numpy as np

for q2 in (1.34,):
 for q4 in (-1.4,):
  for q6 in (-1.,0.,1.,2.):
   for q7 in (0.,np.pi/2):
    env=make_env();s,_=env.reset(seed=0,options={"object_count":4});c=s.get_object_from_name("cube3");p0=np.array([s.get(c,f) for f in ("x","y","z")]);bx=.77;bg=np.array([bx,p0[1],np.pi]);qg=np.array([0.,q2,np.pi,q4,0.,q6,q7])
    def go(bg,qg,grip,n):
     global s
     for _ in range(n):
      r=s.get_object_from_name("robot");b=np.array([s.get(r,f) for f in ("pos_base_x","pos_base_y","pos_base_rot")]);q=np.array([s.get(r,f"pos_arm_joint{i}") for i in range(1,8)]);a=np.zeros(11,np.float32);e=bg-b;e[2]=(e[2]+np.pi)%(2*np.pi)-np.pi;a[:3]=np.clip(e,-.1,.1);a[3:10]=np.clip(qg-q,-.1,.1);a[10]=grip;s,_,_,_,_=env.step(a)
    go(np.array([1.,p0[1],np.pi]),qg,1.,75);go(bg,qg,1.,20);go(bg,qg,0.,15);lift=qg.copy();lift[1]=max(.7,q2-.5);go(bg,lift,0.,25)
    c=s.get_object_from_name("cube3");p=np.array([s.get(c,f) for f in ("x","y","z")]);print(q6,round(q7,2),np.round(p-p0,3).tolist());env.close()
