import math,numpy as np
from env_client import make_env
from probe_grasp_structured import command,val
from replay_grasp_route import route
e=make_env();s,info=e.reset(seed=0,options={'object_count':0});print('names',s.get_object_names(),info);tx=val(s,'box0','pose_x');ty=val(s,'box0','pose_y')
for q,r,ang,rot in route[-4:]:
 b=[tx+r*math.cos(ang),ty+r*math.sin(ang),rot];s=command(e,s,b,q,1.,35);s=command(e,s,b,q,-1.,2)
for _ in range(8):
 a=np.zeros(11,np.float32);a[4]=-.2;a[6]=.2;a[10]=-1;s=e.step(a)[0]
for _ in range(30):
 a=np.zeros(11,np.float32);a[0]=np.clip(.55-val(s,'box0','pose_x'),-.2,.2);a[1]=np.clip(-val(s,'box0','pose_y'),-.2,.2);a[4]=.2;a[6]=-.2;a[10]=-1
 s,r,t,tr,i=e.step(a)
 if val(s,'robot','joint_2')>=.82:break
a=np.zeros(11,np.float32);a[10]=1
for k in range(5):s,r,t,tr,i=e.step(a);print(k,r,t,[round(val(s,'box0','pose_'+c),3) for c in 'xyz'])
e.close()
