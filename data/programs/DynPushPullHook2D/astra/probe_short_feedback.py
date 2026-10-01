import numpy as np,math
from env_client import make_env
for seed in [3,4,6,11,12,21,23,26,27]:
 e=make_env();s,i=e.reset(seed=seed);rb=s.get_object_from_name('robot');h=s.get_object_from_name('hook')
 for k in range(250):
  angle=s.get(h,'theta')+math.pi/2;u=np.array([math.cos(angle),math.sin(angle)]);v=np.array([-u[1],u[0]])
  tip=np.array([s.get(h,'x'),s.get(h,'y')])-s.get(h,'length_side2')*u+s.get(h,'width')*.5*v
  goal=np.clip(tip-.28*u,[.25,.25],[3.24,1.46])
  delta=goal-np.array([s.get(rb,'x'),s.get(rb,'y')]);dt=(angle-s.get(rb,'theta')+math.pi)%(2*math.pi)-math.pi
  grip=-.019 if np.linalg.norm(delta)<.025 and abs(dt)<.03 else .019
  s,r,d,tr,i=e.step(np.array([*np.clip(delta,-.049,.049),np.clip(dt,-.064,.064),-.099,grip],dtype=np.float32))
  if s.get(h,'held'):break
 print(seed,k+1,'hook',*[round(s.get(h,f),3) for f in ['x','y','theta','held']],flush=True)
 e.close()
