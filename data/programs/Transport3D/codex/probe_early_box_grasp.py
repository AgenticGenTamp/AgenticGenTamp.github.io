"""Close throughout the final waypoint transition to vary grasp contact height."""
import math,numpy as np
from env_client import make_env
from probe_grasp_structured import command,val
from replay_grasp_route import route
e=make_env();s,_=e.reset(seed=0,options={'object_count':0});tx=val(s,'box0','pose_x');ty=val(s,'box0','pose_y')
for idx in [24,26]:
 q,r,ang,rot=route[idx];b=[tx+r*math.cos(ang),ty+r*math.sin(ang),rot]
 s=command(e,s,b,q,1.,35);s=command(e,s,b,q,-1.,2)
q,r,ang,rot=route[27];b=np.array([tx+r*math.cos(ang),ty+r*math.sin(ang),rot])
for k in range(40):
 cur=np.array([val(s,'robot',f) for f in ('pos_base_x','pos_base_y','pos_base_rot')]);cq=np.array([val(s,'robot','joint_%d'%i) for i in range(1,8)])
 a=np.zeros(11,np.float32);a[:3]=np.clip(b-cur,-.2,.2);a[3:10]=np.clip(q-cq,-.2,.2);a[10]=-1;s=e.step(a)[0]
 if val(s,'robot','grasp_active')>.5:
  print('HIT step',k,'q',[val(s,'robot','joint_%d'%i) for i in range(1,8)],'tf',[val(s,'robot','grasp_tf_'+c) for c in 'xyz']);break
else:print('MISS')
e.close()
