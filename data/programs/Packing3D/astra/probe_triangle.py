from env_client import make_env
import numpy as np
E=make_env()
q=[0,.391676098,-np.pi,-2.05185294,0,-.452491313,np.pi/2]
for mode in ['simultaneous','arrived','openfirst']:
 s,info=E.reset(seed=0)
 p=s.get_object_from_name('part1'); robot=s.get_object_from_name('robot')
 dest=np.array([s.get(p,'pose_x')-.53265405,s.get(p,'pose_y')-.001,0]+q)
 if mode=='openfirst':
  a=np.zeros(11);a[10]=1;s,*_=E.step(a)
 for i in range(15):
  cur=np.array([s.get(robot,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']+[f'joint_{j}' for j in range(1,8)]])
  a=np.r_[np.clip(dest-cur,-.2,.2),-1 if mode!='arrived' or np.max(abs(dest-cur))<1e-5 else 1]
  s,r,t,tr,inf=E.step(a)
  if s.get(robot,'grasp_active'):
   print(mode,'SUCCESS step',i,'robot',dict(zip(E.observation_space.type_features[robot.type],s.data[robot])),flush=True);break
 else:print(mode,'FAIL',s.data[robot],flush=True)
E.close()
