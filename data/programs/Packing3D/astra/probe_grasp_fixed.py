from env_client import make_env
import numpy as np
E=make_env()
q=[0,.391676098,-np.pi,-2.05185294,0,-.452491313,np.pi/2]
for seed in range(10):
 s,info=E.reset(seed=seed)
 p=s.get_object_from_name('part0'); robot=s.get_object_from_name('robot')
 dest=np.array([s.get(p,'pose_x')-.53265405,s.get(p,'pose_y')-.001,0]+q)
 for i in range(15):
  cur=np.array([s.get(robot,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']+[f'joint_{j}' for j in range(1,8)]])
  a=np.r_[np.clip(dest-cur,-.2,.2),-1]
  s,r,t,tr,inf=E.step(a)
  if s.get(robot,'grasp_active'):break
 print(seed,p.type.name,'grasp',s.get(robot,'grasp_active'),'step',i,'base',[s.get(robot,f) for f in ['pos_base_x','pos_base_y']],'tf',[s.get(robot,f'grasp_tf_{v}') for v in 'xyz'],flush=True)
E.close()
