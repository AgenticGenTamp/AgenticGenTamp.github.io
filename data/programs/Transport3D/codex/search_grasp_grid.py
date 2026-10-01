import sys
import numpy as np
from env_client import make_env

q2 = float(sys.argv[1])
q4 = float(sys.argv[2]) if len(sys.argv) > 2 else -2.5
q6 = float(sys.argv[3]) if len(sys.argv) > 3 else -0.87
env=make_env(); s,info=env.reset(seed=1)
def g(n,f): return float(s.get(s.get_object_from_name(n),f))
target='box0'
tx,ty=g(target,'pose_x'),g(target,'pose_y')
def step(target_x,target_y, grip=0):
 global s
 a=np.zeros(11,np.float32); a[0]=np.clip(target_x-g('robot','pos_base_x'),-.2,.2); a[1]=np.clip(target_y-g('robot','pos_base_y'),-.2,.2); a[4]=np.clip(q2-g('robot','joint_2'),-.2,.2); a[6]=np.clip(q4-g('robot','joint_4'),-.2,.2); a[8]=np.clip(q6-g('robot','joint_6'),-.2,.2); a[10]=grip
 s,*_=env.step(a)
 return g('robot','grasp_active')>0.5
# establish elbow target and open
for _ in range(12): step(g('robot','pos_base_x'),g('robot','pos_base_y'),1)
offs=np.arange(-.7,.701,.1)
for iy,oy in enumerate(offs):
 xs=offs if iy%2==0 else offs[::-1]
 for ox in xs:
  bx,by=tx+ox,ty+oy
  while abs(bx-g('robot','pos_base_x'))>.01 or abs(by-g('robot','pos_base_y'))>.01: step(bx,by,1)
  step(bx,by,1); hit=step(bx,by,-1)
  if hit:
   print('HIT',q2,'offset',ox,oy,'q',[g('robot',f'joint_{i}') for i in range(1,8)],'tf',[g('robot',f'grasp_tf_{c}') for c in 'xyz']); env.close(); sys.exit()
print('MISS',q2); env.close()
