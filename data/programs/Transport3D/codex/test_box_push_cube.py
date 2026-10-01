import math,numpy as np
from env_client import make_env
from probe_grasp_structured import command,val
from replay_grasp_route import route
e=make_env();s,_=e.reset(seed=0);tx,ty=val(s,'box0','pose_x'),val(s,'box0','pose_y')
for idx in [24,26,27]:
 q,r,ang,rot=route[idx];b=[tx+r*math.cos(ang),ty+r*math.sin(ang),rot];s=command(e,s,b,q,1.,35);s=command(e,s,b,q,-1.,2)
cx,cy=val(s,'cube0','pose_x'),val(s,'cube0','pose_y');before=[cx,cy,val(s,'cube0','pose_z')]
for _ in range(20):
 a=np.zeros(11,np.float32);a[0]=np.clip(cx-val(s,'box0','pose_x'),-.2,.2);a[1]=np.clip(cy-val(s,'box0','pose_y'),-.2,.2);a[10]=-1;s=e.step(a)[0]
print('cube',before,[val(s,'cube0','pose_'+c) for c in 'xyz'],'box',[val(s,'box0','pose_'+c) for c in 'xyz'])
e.close()
