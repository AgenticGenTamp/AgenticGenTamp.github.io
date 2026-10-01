from env_client import make_env
import numpy as np
E=make_env();q=[0,.391676098,-np.pi,-2.05185294,0,-.452491313,np.pi/2]
for seed in range(10):
 s,info=E.reset(seed=seed)
 tris=[o.name for o in s.get_objects(E.observation_space.get_type('Kinematic3DTriangle'))]
 for name in tris:
  s,info=E.reset(seed=seed);p=s.get_object_from_name(name);robot=s.get_object_from_name('robot');typ=s.get(p,'triangle_type');off=.02 if typ==1 else 0
  dest=np.array([s.get(p,'pose_x')-.53265405+off,s.get(p,'pose_y')-.001+off,0]+q)
  for i in range(8):
   cur=np.array([s.get(robot,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']+[f'joint_{j}' for j in range(1,8)]])
   a=np.r_[np.clip(dest-cur,-.2,.2),-1];s,*_=E.step(a)
   if s.get(robot,'grasp_active'):break
  print(seed,name,'type',typ,'grasp',s.get(robot,'grasp_active'),'steps',i+1,flush=True)
E.close()
