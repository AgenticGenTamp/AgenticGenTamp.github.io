from env_client import make_env
from kinematics import solve,fk
import numpy as np,math,time
rng=np.random.default_rng(7)
e=make_env();s,_=e.reset(seed=14)
def pos(name):
 o=s.get_object_from_name(name);return np.array([s.get(o,'pose_'+f) for f in 'xyz'])
g=pos('green0');b=pos('blocker');d=(g-b)/np.linalg.norm(g-b);yaw=math.atan2(d[1],d[0]);e.close();best=1;start=time.time()
for xy in [[x,y] for x in [3.78,3.74,3.7] for y in [0,.2,-.2,.4]]+[[x,y] for x in [4.98,4.9,4.7] for y in [1.0,1.02,1.08]]:
 for off in [-.035,0,.035]:
  for j in range(6):
   ini=np.array([*xy,-math.pi/2,rng.uniform(.5,1.39),rng.uniform(-.3,1),rng.uniform(-.7,3.8),rng.uniform(-2,0),rng.uniform(-math.pi,math.pi),rng.uniform(-2,0),rng.uniform(-math.pi,math.pi)])
   cr,er=solve(b+off*d+[0,0,.03],yaw,xy,ini)
   cg,eg=solve(g-.035*d+[0,0,.03],yaw,xy,cr)
   if max(er,eg)<best:
    best=max(er,eg);print('BEST',xy,off,j,er,eg,'cr',cr.tolist(),'cg',cg.tolist(),flush=True)
   if max(er,eg)<.005:print('FOUND',flush=True);quit()
print('TIME',time.time()-start)
