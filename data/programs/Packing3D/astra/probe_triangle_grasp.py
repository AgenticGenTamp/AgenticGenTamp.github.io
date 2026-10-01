from env_client import make_env
import numpy as np
E=make_env()
for seed in range(8):
 s,info=E.reset(seed=seed,options={'object_count':1});rob=s.get_object_from_name('robot');p=s.get_object_from_name('part0')
 fs=['pos_base_x','pos_base_y','pos_base_rot']+[f'joint_{i}' for i in range(1,8)]
 pos=np.array([s.get(p,k) for k in ['pose_x','pose_y','pose_z']]);target=np.array([pos[0]-.53265405,pos[1]-.001,0,0,.391676098,-np.pi,-2.05185294,0,-.452491313,np.pi/2])
 for k in range(12):
  a=np.zeros(11);a[:10]=np.clip(target-np.array([s.get(rob,k) for k in fs]),-.2,.2);s,*_=E.step(a)
 a=np.zeros(11);a[10]=-1;s,*_=E.step(a)
 print(seed,'triangle_type',s.get(p,'triangle_type'),'pos',pos,'held',s.get(rob,'grasp_active'),'err',np.array([s.get(rob,k) for k in fs])-target,flush=True)
E.close()
