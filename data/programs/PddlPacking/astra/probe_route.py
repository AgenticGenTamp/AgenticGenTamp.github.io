import numpy as np
from env_client import make_env
from kinematics import Q0

e=make_env();f=['base_x','base_y','base_rot']+['joint_'+str(i) for i in range(1,8)]
for edge_abs in [1.0,1.05,1.1,1.15]:
 for seed in range(10):
  for sign in [-1,1]:
   s,_=e.reset(seed=seed);r=s.get_object_from_name('robot');steps=0
   def v():return np.array([s.get(r,k) for k in f])
   def move(base):
    global s,steps
    t=np.r_[base,Q0]
    for _ in range(30):
     prev=v();d=t-prev
     for j in [2,7,9]:d[j]=(d[j]+np.pi)%(2*np.pi)-np.pi
     if np.max(abs(d))<.001:return True
     s,*_=e.step(np.r_[np.clip(d,-.2,.2),1]);steps+=1
     if np.max(abs(v()-prev))<1e-5:return False
    return False
   good=move([-.8,.5*sign,0])
   for waypoint in [[-.85,sign*edge_abs,sign*np.pi/2],[.8,sign*edge_abs,sign*np.pi],[.8,-.5*sign,sign*np.pi]]:
    good=move(waypoint) and good
    if not good:break
   if not good:print('FAIL',edge_abs,seed,sign,'q',v()[:3],flush=True)
   elif seed==0:print('PASS',edge_abs,sign,steps,flush=True)
e.close()
