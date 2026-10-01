import numpy as np
from env_client import make_env
from kinematics import Q0

e=make_env();f=['base_x','base_y','base_rot']+['joint_'+str(i) for i in range(1,8)]
for seed in [0,1,42]:
 for ys in [-.7,0,.7]:
  for sign in [-1,1]:
   s,_=e.reset(seed=seed);r=s.get_object_from_name('robot');steps=0
   def v():return np.array([s.get(r,k) for k in f])
   def move(base):
    global s,steps
    t=np.r_[base,Q0]
    for _ in range(40):
     prev=v();d=t-prev
     for j in [2,7,9]:d[j]=(d[j]+np.pi)%(2*np.pi)-np.pi
     if np.max(abs(d))<.001:return True
     s,*_=e.step(np.r_[np.clip(d,-.2,.2),1]);steps+=1
     if np.max(abs(v()-prev))<1e-5:return False
    return False
   route=[[-.85,1.15,np.pi/2],[.85,1.15,np.pi],[.66,ys,np.pi],[.85,sign*1.15,sign*np.pi/2],[-.66,sign*1.15,0],[-.66,-ys,0]]
   good=True
   for w in route:
    if not move(w):good=False;break
   print('PASS' if good else 'FAIL',seed,ys,sign,steps,'final',v()[:3].round(3),flush=True)
e.close()
