import numpy as np
from env_client import make_env
from kinematics import ik,fk

e=make_env();s,_=e.reset(seed=0);r=s.get_object_from_name('robot')
f=['base_x','base_y','base_rot']+['joint_'+str(i) for i in range(1,8)]
def v():return np.array([s.get(r,k) for k in f])
def act(t):
 global s
 for _ in range(30):
  last=v();a=np.zeros(11);a[:10]=np.clip(t-last,-.1,.1);a[10]=1;s,*_=e.step(a.astype(np.float32))
  if np.linalg.norm(t-v())<1e-5 or np.linalg.norm(v()-last)<1e-5:break
 a=np.zeros(11);a[10]=-1;s,*_=e.step(a.astype(np.float32))
 if s.get(r,'grasp_active'):
  print('SUCCESS',v().tolist(),'tf',[s.get(r,'grasp_tf_'+k) for k in ['x','y','z','qx','qy','qz','qw']],flush=True);e.close();raise SystemExit
for z in np.arange(.95,.55,-.01):
 q=ik(np.array([.72,.25,z]),q0=v()[3:]);t=np.r_[-.76530257,.050070107,0,q];act(t)
 print('z',z,'err',np.linalg.norm(t-v()),'fk',fk(v()[3:])[0].tolist(),flush=True)
e.close()
