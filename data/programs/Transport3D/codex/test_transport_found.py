import math, numpy as np
from env_client import make_env
from probe_grasp_structured import command,val
from replay_grasp_route import route

env=make_env();s,_=env.reset(seed=0);tx=val(s,'box0','pose_x');ty=val(s,'box0','pose_y')
for q,r,ang,rot in route[-4:]:
 b=[tx+r*math.cos(ang),ty+r*math.sin(ang),rot]
 s=command(env,s,b,q,1.,35);s=command(env,s,b,q,-1.,2)
print('held',val(s,'robot','grasp_active'),'z',val(s,'box0','pose_z'))
# Try shoulder retraction while maintaining all other joint positions.
for k in range(8):
 a=np.zeros(11,np.float32);a[4]=-.2;a[10]=-1;s=env.step(a)[0]
 print('lift',k,round(val(s,'robot','joint_2'),3),round(val(s,'box0','pose_z'),3))
# Carry by exact object feedback to above table center.
for k in range(15):
 a=np.zeros(11,np.float32);a[0]=np.clip(.6-val(s,'box0','pose_x'),-.2,.2);a[1]=np.clip(0-val(s,'box0','pose_y'),-.2,.2);a[10]=-1;s=env.step(a)[0]
print('carried',*[round(val(s,'box0','pose_'+c),3) for c in 'xyz'])
for k in range(8):
 a=np.zeros(11,np.float32);a[4]=.2;a[10]=-1;s=env.step(a)[0]
 print('lower',k,round(val(s,'robot','joint_2'),3),round(val(s,'box0','pose_z'),3))
a=np.zeros(11,np.float32);a[10]=1
for _ in range(5):s=env.step(a)[0]
print('released',val(s,'robot','grasp_active'),*[round(val(s,'box0','pose_'+c),3) for c in 'xyz'])
env.close()
