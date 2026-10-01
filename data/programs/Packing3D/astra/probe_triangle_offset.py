from env_client import make_env
import numpy as np
E=make_env();s,info=E.reset(seed=0,options={'object_count':1});rob=s.get_object_from_name('robot');p=s.get_object_from_name('part0')
fs=['pos_base_x','pos_base_y','pos_base_rot']+[f'joint_{i}' for i in range(1,8)]
pos=np.array([s.get(p,k) for k in ['pose_x','pose_y','pose_z']]);baseline=np.array([pos[0]-.53265405,pos[1]-.001,0,0,.391676098,-np.pi,-2.05185294,0,-.452491313,np.pi/2])
for dx in [0,-.02,.02,-.04,.04,-.06,.06]:
 for dy in [0,-.02,.02,-.04,.04,-.06,.06]:
  target=baseline.copy();target[0]+=dx;target[1]+=dy
  for k in range(6):
   a=np.zeros(11);a[:10]=np.clip(target-np.array([s.get(rob,k) for k in fs]),-.2,.2);s,*_=E.step(a)
  a=np.zeros(11);a[10]=-1;s,*_=E.step(a)
  if s.get(rob,'grasp_active'):
   print('GRASP',dx,dy,'config',{k:s.get(rob,k) for k in E.observation_space.type_features[rob.type]},flush=True);E.close();quit()
 print('dx complete',dx,flush=True)
print('NONE');E.close()
