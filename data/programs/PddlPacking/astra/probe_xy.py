import numpy as np
from env_client import make_env
from kinematics import ik

e=make_env();s,_=e.reset(seed=0);r=s.get_object_from_name('robot')
f=['base_x','base_y','base_rot']+['joint_'+str(i) for i in range(1,8)]
def v():return np.array([s.get(r,k) for k in f])
def act(t):
 global s
 for _ in range(30):
  last=v();a=np.zeros(11);a[:10]=np.clip(t-last,-.15,.15);a[10]=1;s,*_=e.step(a.astype(np.float32))
  if np.linalg.norm(t-v())<1e-5 or np.linalg.norm(v()-last)<1e-5:break
 a=np.zeros(11);a[10]=-1;s,*_=e.step(a.astype(np.float32))
 if s.get(r,'grasp_active'):
  print('SUCCESS',v().tolist(),'tf',[s.get(r,'grasp_tf_'+k) for k in ['x','y','z','qx','qy','qz','qw']],flush=True);e.close();raise SystemExit
q=ik(np.array([.72,.25,.8]),h=.94)
t=np.r_[-1,.05,0,q];act(t)
for iy,y in enumerate(np.arange(-.10,.251,.02)):
 for x in np.arange(-.9,-.649,.02)[::1 if iy%2==0 else -1]:
  t[:2]=[x,y];act(t)
 print('y',y,'state',v().tolist(),flush=True)
e.close()
