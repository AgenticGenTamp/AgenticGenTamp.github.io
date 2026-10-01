"""Test empirically shortened arm posture on east-side blocker failures."""
import math, sys
import numpy as np
from env_client import make_env

Q=np.array([.75099605,.64684612,1.00101125,-2.32130003,-2.48106384,-2.09400010,-2.10809493])
OFF=np.array([.49231529,.40115744])
def g(s,n,f):return float(s.get(s.get_object_from_name(n),f))
seed=int(sys.argv[1]);e=make_env()
for theta in np.arange(-math.pi,math.pi,.03):
 s,_=e.reset(seed=seed);b=np.array([g(s,'blocker','pose_x'),g(s,'blocker','pose_y')])
 c,z=math.cos(theta),math.sin(theta);t=b-np.array([c*OFF[0]-z*OFF[1],z*OFF[0]+c*OFF[1]])
 if t[0]>5 or t[0]<-1:continue
 side=1.6 if t[1]>=0 else -1.6
 for wi,w in enumerate(([np.array([3.4,side]),np.array([t[0],side]),t] if t[0]>3.65 else [t])):
  for _ in range(30):
   a=np.zeros(11,np.float32);xy=np.array([g(s,'robot','base_x'),g(s,'robot','base_y')]);a[:2]=np.clip(w-xy,-.2,.2)
   a[2]=np.clip((theta-g(s,'robot','base_rot')+math.pi)%(2*math.pi)-math.pi,-.2,.2)
   # Fold the arm at the outside routing waypoint, before approaching the table.
   if wi>=0:
    for j in range(7):
     d=Q[j]-g(s,'robot','joint_'+str(j+1));d=(d+math.pi)%(2*math.pi)-math.pi if j in (4,6) else d;a[3+j]=np.clip(d,-.2,.2)
   a[10]=1;s,*_=e.step(a)
 a=np.zeros(11,np.float32);a[10]=-1;s,*_=e.step(a)
 if g(s,'blocker','grasp_active')>.5:
  print('SHORT_HIT',seed,'theta',theta,'base',t.tolist(),'q',Q.tolist());break
else:print('SHORT_MISS',seed)
e.close()
