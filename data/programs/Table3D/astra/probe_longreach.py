from env_client import make_env
import numpy as np,time
E=make_env()
fields=['pos_base_x','pos_base_y','pos_base_rot']+['joint_'+str(i) for i in range(1,8)]
def vals(s,r):return np.array([s.get(r,f) for f in fields])
def move(s,r,t,grip=0):
 for _ in range(30):
  d=t-vals(s,r)
  if max(abs(d))<1e-4 and grip==0:return s,False
  prev=vals(s,r);s,re,te,tr,info=E.step(np.r_[np.clip(d,-.4,.4),grip])
  if (grip < 0 and s.get(r,'grasp_active')) or te:return s,te
  if max(abs(vals(s,r)-prev))<1e-5:return s,te
 return s,False
for seed in range(100):
 s,_=E.reset(seed=seed);r=next(iter(s.get_objects(E.observation_space.get_type('Kinematic3DRobot'))))
 cubes=list(s.get_objects(E.observation_space.get_type('Kinematic3DCuboid')));cubes=[c for c in cubes if c.name.startswith('cube')];c=min(cubes,key=lambda c:s.get(c,'pose_x'))
 target=vals(s,r);target[:2]=[s.get(c,'pose_x')-.61,s.get(c,'pose_y')]
 # adjust arm away from table before base approach
 target[0]=-.3;target[3:]=[0,.7,-np.pi,-1.7,0,.7+1.7-np.pi,np.pi/2];s,_=move(s,r,target)
 target[0]=s.get(c,'pose_x')-.61;s,_=move(s,r,target,-1)
 grasp=s.get(r,'grasp_active');result=False
 if grasp:
  target=vals(s,r);target[4]-=.5;s,result=move(s,r,target)
 print(seed,len(cubes),round(s.get(c,'pose_x'),3),bool(grasp),result,vals(s,r).round(3).tolist(),flush=True)
E.close()
