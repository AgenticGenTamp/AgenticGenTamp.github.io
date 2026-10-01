from env_client import make_env
import numpy as np,time
E=make_env(); fields=['pos_base_x','pos_base_y','pos_base_rot']+['joint_'+str(i) for i in range(1,8)]
def vals(s,r):return np.array([s.get(r,f) for f in fields])
N=0; found=[]
for q4 in [-2.7,-2.5,-2.3,-2.1,-1.9,-1.7,-1.5,-1.3,-1.1]:
 for q2 in np.arange(-1,.81,.1):
  s,_=E.reset(seed=0);r=s.get_objects(E.observation_space.get_type('Kinematic3DRobot'))[0]
  cubes=[c for c in s.get_objects(E.observation_space.get_type('Kinematic3DCuboid')) if c.name.startswith('cube')];c=min(cubes,key=lambda c:s.get(c,'pose_x'));cx=s.get(c,'pose_x');cy=s.get(c,'pose_y')
  for x in [-.4,-.7]+list(np.arange(-.7,.251,.025)):
   target=np.array([x,cy,0,0,q2,-np.pi,q4,0,q2-q4-np.pi,np.pi/2]);a=np.r_[np.clip(target-vals(s,r),-.4,.4),-1]
   s,re,te,tr,info=E.step(a);N+=1
   if s.get(r,'grasp_active'):
    p=vals(s,r);row=[round(float(v),5) for v in [q2,q4,cx-p[0]]];found.append(row)
    print('GRASP',row,'qactual',p[3:].round(5).tolist(),'base',p[:3].round(5).tolist(),flush=True);break
 print('Q4DONE',q4,N,flush=True)
print('ALL',found,flush=True);E.close()
