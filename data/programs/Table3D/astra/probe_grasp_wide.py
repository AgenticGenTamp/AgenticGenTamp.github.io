from env_client import make_env
import numpy as np,time
E=make_env(); N=0;t=time.time()

def vals(s,r): return np.array([s.get(r,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']+['joint_'+str(i) for i in range(1,8)]])
def move(s,r,target):
 global N
 for k in range(15):
  d=target-vals(s,r)
  if np.max(np.abs(d))<.001: break
  a=np.r_[np.clip(d,-.4,.4),-1.]
  prev=vals(s,r);s,re,te,tr,inf=E.step(a);N+=1
  if s.get(r,'grasp_active'):
   print('GRASP',N,vals(s,r).tolist(),flush=True)
   print({f:s.get(r,f) for f in E.observation_space.type_features[r.type]},flush=True)
   raise SystemExit
  if np.max(np.abs(vals(s,r)-prev))<.001: break
 return s
for q2 in [-1.2,-.95,-.7,-.45,-.2,.05,.3,.55,.8]:
 for yy in [-.2,-.1,0,.1,.2]:
  q4=-2.5
  s,_=E.reset(seed=0,options={'object_count':1});r=next(iter(s.get_objects(E.observation_space.get_type('Kinematic3DRobot'))))
  cubes=[c for c in s.get_objects(E.observation_space.get_type('Kinematic3DCuboid')) if c.name.startswith('cube')]; c=cubes[0]
  cy=s.get(c,'pose_y'); cx=s.get(c,'pose_x')
  target=vals(s,r);target[:2]=[-.8,cy+yy]; s=move(s,r,target)
  target[4]=q2;target[6]=q4;s=move(s,r,target)
  for x in np.arange(-.8,.31,.025):
   target[0]=x;s=move(s,r,target)
  print('ROW',q2,yy,N,round(time.time()-t,1),vals(s,r)[:2].tolist(),flush=True)
E.close()
