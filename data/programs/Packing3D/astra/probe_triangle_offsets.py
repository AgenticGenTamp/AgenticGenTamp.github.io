from env_client import make_env
import numpy as np
E=make_env();s,info=E.reset(seed=0);p=s.get_object_from_name('part1'); robot=s.get_object_from_name('robot')
px=s.get(p,'pose_x');py=s.get(p,'pose_y')
q=[0,.391676098,-np.pi,-2.05185294,0,-.452491313,np.pi/2]
for oy in [0,-.02,.02,-.04,.04,-.06,.06]:
 for ox in [0,-.02,.02,-.04,.04,-.06,.06]:
  dest=np.array([px-.53265405+ox,py-.001+oy,0]+q)
  for i in range(6):
   cur=np.array([s.get(robot,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']+[f'joint_{j}' for j in range(1,8)]])
   a=np.r_[np.clip(dest-cur,-.2,.2),-1]
   s,*_=E.step(a)
   if s.get(robot,'grasp_active'):
    print('SUCCESS offsets',ox,oy,'robot',dict(zip(E.observation_space.type_features[robot.type],s.data[robot])),flush=True);E.close();raise SystemExit
   if np.max(abs(dest-cur))<1e-5:break
print('FAIL',flush=True)
E.close()
