"""Adapt the box contact pose downward by its measured local Jacobian."""
import math,numpy as np
from env_client import make_env
from probe_grasp_structured import command,val
from replay_grasp_route import route

e=make_env();s,_=e.reset(seed=0);target='cube0';tx=val(s,target,'pose_x');ty=val(s,target,'pose_y')
for idx in [24,26,27]:
 q,r,ang,rot=route[idx];b=[tx+r*math.cos(ang),ty+r*math.sin(ang),rot]
 s=command(e,s,b,q,1.,35)
# Inverse of the first local Cartesian lift step, repeated to lower ~75 mm.
dq=-np.array([-.008,0.,.029,-.045,-.003,.107,.062])
a=np.zeros(11,np.float32);a[3:10]=dq;a[0]=-.043;a[1]=-.024;a[10]=1
s=e.step(a)[0];s=e.step(a)[0]
bx0=val(s,'robot','pos_base_x');by0=val(s,'robot','pos_base_y')
for oy in np.arange(-.12,.121,.02):
 for ox in np.arange(-.12,.121,.02):
  for grip in [1.,-1.]:
   a=np.zeros(11,np.float32);a[0]=np.clip(bx0+ox-val(s,'robot','pos_base_x'),-.2,.2);a[1]=np.clip(by0+oy-val(s,'robot','pos_base_y'),-.2,.2);a[10]=grip;s=e.step(a)[0]
  if val(s,'robot','grasp_active')>.5:
   print('HIT',ox,oy,'q',[val(s,'robot','joint_%d'%i) for i in range(1,8)],'baseoff',val(s,'robot','pos_base_x')-tx,val(s,'robot','pos_base_y')-ty);e.close();raise SystemExit
print('MISS');e.close()
