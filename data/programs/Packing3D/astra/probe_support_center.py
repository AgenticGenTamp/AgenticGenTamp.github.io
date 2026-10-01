from env_client import make_env
import numpy as np
E=make_env()
for z in [.085,.09,.095,.1,.105,.11]:
 s,info=E.reset(seed=1,options={'object_count':1});rob=s.get_object_from_name('robot');p=s.get_object_from_name('part0');rack=s.get_object_from_name('rack')
 features=['pos_base_x','pos_base_y','pos_base_rot']+[f'joint_{i}' for i in range(1,8)]
 def pp():return np.array([s.get(p,k) for k in ['pose_x','pose_y','pose_z']])
 def act(a):
  global s
  s,r,te,tr,info=E.step(a);return te
 target=np.array([pp()[0]-.53265405,pp()[1]-.001,0,0,.391676098,-np.pi,-2.05185294,0,-.452491313,np.pi/2])
 for k in range(12):
  a=np.zeros(11);a[:10]=np.clip(target-np.array([s.get(rob,k) for k in features]),-.2,.2);act(a)
 a=np.zeros(11);a[10]=-1;act(a)
 a=np.zeros(11);a[4]=-.15;a[8]=-.15;act(a)
 for k in range(5):
  a=np.zeros(11);a[:2]=np.clip(np.array([.30,0])-pp()[:2],-.2,.2);act(a)
 for k in range(20):
  dz=pp()[2]-z
  if abs(dz)<.000002:break
  a=np.zeros(11);a[4]=np.clip(dz/.35,-.03,.03);a[8]=a[4];act(a)
 a=np.zeros(11);a[10]=1;te=act(a)
 print('RESULT',z,'count',info,'type',p.type.name,'pose',pp(),'held',s.get(rob,'grasp_active'),'done',te,flush=True)
E.close()
