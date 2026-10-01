"""Try to use the held box as a kinematic elevator for a cube."""
import math,numpy as np
from env_client import make_env
from probe_grasp_structured import command,val
from replay_grasp_route import route
e=make_env();s,_=e.reset(seed=0);tx,ty=val(s,'box0','pose_x'),val(s,'box0','pose_y')
for idx in [24,26,27]:
 q,r,ang,rot=route[idx];q=q.copy()
 if idx==27:q[0]+=.3
 b=[tx+r*math.cos(ang),ty+r*math.sin(ang),rot];s=command(e,s,b,q,1.,35);s=command(e,s,b,q,-1.,2)
for k in range(9):
 cx,cy=val(s,'cube0','pose_x'),val(s,'cube0','pose_y');bx,by=val(s,'robot','pos_base_x'),val(s,'robot','pos_base_y')
 a=np.zeros(11,np.float32);a[0]=np.clip(bx+cx-val(s,'box0','pose_x'),-.2,.2);a[1]=np.clip(by+cy-val(s,'box0','pose_y'),-.2,.2);a[4]=-.2 if k>5 else 0;a[10]=-1;s=e.step(a)[0]
 print(k,'cube',[round(val(s,'cube0','pose_'+c),3) for c in 'xyz'],'box',[round(val(s,'box0','pose_'+c),3) for c in 'xyz'])
# Small final lift keeps contact while clearing the table edge.
cx,cy=val(s,'cube0','pose_x'),val(s,'cube0','pose_y');bx,by=val(s,'robot','pos_base_x'),val(s,'robot','pos_base_y')
a=np.zeros(11,np.float32);a[0]=np.clip(bx+cx-val(s,'box0','pose_x'),-.2,.2);a[1]=np.clip(by+cy-val(s,'box0','pose_y'),-.2,.2);a[4]=.01;a[10]=-1;s=e.step(a)[0]
print('trim','cube',[round(val(s,'cube0','pose_'+c),3) for c in 'xyz'],'box',[round(val(s,'box0','pose_'+c),3) for c in 'xyz'])
for gx,gy,n in [(0.,-1.,25),(0.,0.,12),(.4,0.,5)]:
 for k in range(n):
  cx,cy=val(s,'cube0','pose_x'),val(s,'cube0','pose_y');bx,by=val(s,'robot','pos_base_x'),val(s,'robot','pos_base_y')
  a=np.zeros(11,np.float32);a[0]=np.clip(gx-cx,-.2,.2);a[1]=np.clip(gy-cy,-.2,.2);a[10]=-1;s,r,t,tr,i=e.step(a)
 print('stage',gx,gy,'cube',[round(val(s,'cube0','pose_'+c),3) for c in 'xyz'],'box',[round(val(s,'box0','pose_'+c),3) for c in 'xyz'],t)
# At the lip, lift the pusher so the cube climbs over the table edge.
a=np.zeros(11,np.float32);a[4]=-.12;a[10]=-1;s=e.step(a)[0]
print('lip-lift',[round(val(s,'cube0','pose_'+c),3) for c in 'xyz'])
for k in range(4):
 a=np.zeros(11,np.float32);a[4]=.12;a[10]=-1;s,r,t,tr,i=e.step(a)
print('final cube',[round(val(s,'cube0','pose_'+c),3) for c in 'xyz'],'box',[round(val(s,'box0','pose_'+c),3) for c in 'xyz'],t)
e.close()
