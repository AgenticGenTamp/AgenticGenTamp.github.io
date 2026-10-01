from env_client import make_env
import numpy as np
E=make_env();q=[0,.391676098,-np.pi,-2.05185294,0,-.452491313,np.pi/2]
for seed,name in [(0,'part1'),(1,'part0')]:
 s,info=E.reset(seed=seed);p=s.get_object_from_name(name);robot=s.get_object_from_name('robot');px=s.get(p,'pose_x');py=s.get(p,'pose_y')
 print('TYPE',s.get(p,'triangle_type'),flush=True)
 for oy in np.arange(-.08,.121,.02):
  row=[]
  for ox in np.arange(-.08,.121,.02):
   a=np.zeros(11);a[10]=1;s,*_=E.step(a)
   dest=np.array([px-.53265405+ox,py-.001+oy,0]+q)
   for i in range(6):
    cur=np.array([s.get(robot,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']+[f'joint_{j}' for j in range(1,8)]])
    a=np.r_[np.clip(dest-cur,-.2,.2),1];s,*_=E.step(a)
    if np.max(abs(dest-cur))<1e-5:break
   a=np.zeros(11);a[10]=-1;s,*_=E.step(a)
   row.append('X' if s.get(p,'grasp_active') else '.')
  print(round(oy,2),''.join(row),flush=True)
E.close()
