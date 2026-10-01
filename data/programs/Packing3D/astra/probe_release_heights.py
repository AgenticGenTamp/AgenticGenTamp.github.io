from env_client import make_env
import numpy as np
E=make_env()
for seed in [1]:
 s,info=E.reset(seed=seed,options={'object_count':1});rob=s.get_object_from_name('robot');p=s.get_object_from_name('part0');rack=s.get_object_from_name('rack')
 features=['pos_base_x','pos_base_y','pos_base_rot']+[f'joint_{i}' for i in range(1,8)]
 def pp():return np.array([s.get(p,k) for k in ['pose_x','pose_y','pose_z']])
 def act(a):
  global s
  s,r,te,tr,info=E.step(a);return te
 target=np.array([pp()[0]-.53265405,pp()[1]-.001,0,0,.391676098,-np.pi,-2.05185294,0,-.452491313,np.pi/2])
 for k in range(12):
  a=np.zeros(11);a[:10]=np.clip(target-np.array([s.get(rob,k) for k in features]),-.2,.2);act(a)
 a=np.zeros(11);a[10]=-1;act(a);print(seed,'GRASP',s.get(rob,'grasp_active'),pp(),flush=True)
 a=np.zeros(11);a[4]=-.15;a[8]=-.15;act(a);print('LIFT',pp(),flush=True)
 for k in range(5):
  a=np.zeros(11);a[:2]=np.clip(np.array([s.get(rack,'pose_x'),s.get(rack,'pose_y')])-pp()[:2],-.2,.2);act(a)
 print('PLACE',pp(),flush=True)
 for k in range(6):
  dz=pp()[2]-.125
  if abs(dz)<.001:break
  a=np.zeros(11);a[4]=np.clip(dz/.37,-.03,.03);a[8]=a[4];act(a)
 print('LOWER',pp(),flush=True)
 for adjust in [0,-.002,.004,.006,.01,.02,-.05]:
  a=np.zeros(11);a[4]=adjust;a[8]=adjust;act(a)
  a=np.zeros(11);a[10]=1;te=act(a);print('RELEASE',adjust,te,pp(),s.get(rob,'grasp_active'),flush=True)
  if te:break
E.close()
