import numpy as np
from env_client import make_env

env=make_env(); s,_=env.reset(seed=1)
def g(n,f): return float(s.get(s.get_object_from_name(n),f))
def act(a):
 global s
 s,*_=env.step(np.asarray(a,np.float32))
q=np.array([0,-1.6743,-3.8876,-1.5737,.1635,-1.7428,.0253])
target='cube0'; tx,ty=g(target,'pose_x'),g(target,'pose_y')
for oy in np.arange(-.3,.301,.03):
 for ox in np.arange(-.3,.301,.03):
  bx,by=tx-.212+ox,ty-.192+oy
  for _ in range(30):
   a=np.zeros(11);a[0]=np.clip(bx-g('robot','pos_base_x'),-.2,.2);a[1]=np.clip(by-g('robot','pos_base_y'),-.2,.2)
   for j in range(7):a[3+j]=np.clip(q[j]-g('robot',f'joint_{j+1}'),-.2,.2)
   a[10]=1;act(a)
  a=np.zeros(11);a[10]=-1;act(a)
  if g('robot','grasp_active')>.5:
   print('HIT',ox,oy,'base',g('robot','pos_base_x'),g('robot','pos_base_y'),'q',[g('robot',f'joint_{i}') for i in range(1,8)]);quit()
print('MISS')
