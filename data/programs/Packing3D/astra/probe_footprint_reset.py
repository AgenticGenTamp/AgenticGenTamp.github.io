from env_client import make_env
import numpy as np
E=make_env();q=[0,.391676098,-np.pi,-2.05185294,0,-.452491313,np.pi/2]
for seed,name in [(0,'part1'),(1,'part0')]:
 print('SEED',seed,flush=True)
 for ox,oy in [(0,0),(-.04,0),(0,-.04),(-.02,-.02),(.02,.02),(.04,0),(0,.04),(.04,.04),(.06,.02),(.02,.06),(.08,.02),(.02,.08),(.1,.02)]:
  s,info=E.reset(seed=seed);p=s.get_object_from_name(name);robot=s.get_object_from_name('robot');px=s.get(p,'pose_x');py=s.get(p,'pose_y');dest=np.array([px-.53265405+ox,py-.001+oy,0]+q)
  for i in range(6):
   cur=np.array([s.get(robot,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']+[f'joint_{j}' for j in range(1,8)]])
   a=np.r_[np.clip(dest-cur,-.2,.2),-1];s,*_=E.step(a)
   if s.get(p,'grasp_active'):break
  print(ox,oy,int(s.get(p,'grasp_active')),flush=True)
E.close()
