from env_client import make_env
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def trial(lift):
 e=make_env();s,_=e.reset(seed=1);r=s.get_object_from_name('robot');b=s.get_object_from_name('green1');xy=np.array([s.get(b,'pose_x'),s.get(b,'pose_y')]);n=0
 fs=['base_x','base_y','base_rot']+['joint_'+str(i) for i in range(1,8)]
 def goto(v):
  nonlocal s,n
  for _ in range(50):
   p=np.array([s.get(r,f) for f in fs]);d=np.array(v)-p;d[[2,7,9]]=(d[[2,7,9]]+np.pi)%(2*np.pi)-np.pi
   if max(abs(d))<1e-5:return False
   a=np.zeros(11);a[:10]=np.clip(d,-.2,.2);a[10]=-1
   s,_,t,tr,_=e.step(a);n+=1
   if s.get(r,'grasp_active'):
    print('GRASP lift',lift,'step',n,'p',p,'new',[s.get(r,f) for f in fs], 'TF',[s.get(r,'grasp_tf_'+f) for f in ['x','y','z','qx','qy','qz','qw']],flush=True);return True
   if max(abs(p-np.array([s.get(r,f) for f in fs])))<1e-5:return False
  return False
 q=[0,lift,0,-.4,0,.4-lift,0]
 goto([0,-1.5,np.pi]+q);goto([xy[0]+1.3,xy[1]+.188,np.pi]+q)
 for dx in np.arange(1.3,.5,-.025):
  if goto([xy[0]+dx,xy[1]+.188,np.pi]+q): e.close();return True
 print('FAIL',lift, 'last',[round(s.get(r,f),3) for f in fs],flush=True);e.close();return False
for lift in [.55,.4,.25,.7, .85,1.0, .1,-.1]: trial(lift)
