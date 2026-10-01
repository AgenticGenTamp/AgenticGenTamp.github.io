from env_client import make_env
import numpy as np
E=make_env()
seed=0
for seed in range(30):
 s,info=E.reset(seed=seed)
 p=s.get_object_from_name('part0')
 if info['object_count']==1 and 'Cuboid' in p.type.name:break
print('seed',seed,flush=True)
for j4 in np.arange(-3.3,-1.69,.1):
 s,info=E.reset(seed=seed)
 p=s.get_object_from_name('part0'); robot=s.get_object_from_name('robot'); px=s.get(p,'pose_x'); py=s.get(p,'pose_y')
 print('scan',j4,'part',px,py,flush=True)
 for bx in np.arange(-.5,.201,.025):
  for k in range(8):
   a=np.zeros(11); a[0]=np.clip(bx-s.get(robot,'pos_base_x'),-.2,.2); a[1]=np.clip(py-s.get(robot,'pos_base_y'),-.2,.2); a[6]=np.clip(j4-s.get(robot,'joint_4'),-.2,.2);a[10]=-1
   s,r,t,tr,info=E.step(a)
   if s.get(robot,'grasp_active'):
    print('SUCCESS',j4,bx,'robot',dict(zip(E.observation_space.type_features[robot.type],s.data[robot])),flush=True); E.close(); raise SystemExit
   if max(abs(a[0]),abs(a[1]),abs(a[6]))<1e-5:break
E.close()
