import numpy as np,math
from env_client import make_env
for seed in range(6):
 e=make_env();s,i=e.reset(seed=seed)
 rb=s.get_object_from_name('robot');h=s.get_object_from_name('hook')
 ang=s.get(h,'theta');u=np.array([math.cos(ang),math.sin(ang)]);v=np.array([-u[1],u[0]])
 goal=np.array([s.get(h,'x'),s.get(h,'y')])-(s.get(h,'length_side1')-.15)*u-s.get(h,'width')*.5*v-.32*v
 goal=np.clip(goal,[.25,.25],[3.24,1.46]);theta=ang+math.pi/2
 for phase in range(3):
  for k in range(100):
   dx=goal[0]-s.get(rb,'x') if phase>=1 else 0
   dy=goal[1]-s.get(rb,'y') if phase>=2 else 0
   dt=theta-s.get(rb,'theta')
   s,r,d,tr,i=e.step(np.array([np.clip(dx,-.049,.049),np.clip(dy,-.049,.049),np.clip(dt,-.064,.064),-.099,.019],dtype=np.float32))
   if max(abs(dx),abs(dy),abs(dt))<.002:break
 for k in range(15):s,r,d,tr,i=e.step(np.array([0,0,0,0,-.019],dtype=np.float32))
 print(seed,'robot',*[round(s.get(rb,f),3) for f in ['x','y','theta']],'hook',*[round(s.get(h,f),3) for f in ['x','y','theta','held']],flush=True)
 e.close()
