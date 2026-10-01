import numpy as np,math
from env_client import make_env
for seed in range(12):
 e=make_env();s,i=e.reset(seed=seed)
 rb=s.get_objects(e.observation_space.get_type('kin_robot'))[0];h=s.get_objects(e.observation_space.get_type('hook'))[0]
 angle=s.get(h,'theta')+math.pi/2;u=np.array([math.cos(angle),math.sin(angle)]);v=np.array([-u[1],u[0]])
 tip=np.array([s.get(h,'x'),s.get(h,'y')])-s.get(h,'length_side2')*u+s.get(h,'width')*.5*v
 goal=np.clip(tip-.28*u,[.25,.25],[3.24,1.46])
 print('INIT',seed,'robot',*[round(s.get(rb,f),3) for f in ['x','y','theta']],'hook',*[round(s.get(h,f),3) for f in ['x','y','theta']],'tip',tip,'goal',goal,flush=True)
 for k in range(100):
  delta=goal-np.array([s.get(rb,'x'),s.get(rb,'y')]);dt=angle-s.get(rb,'theta')
  s,r,d,tr,i=e.step(np.array([*np.clip(delta,-.049,.049),np.clip(dt,-.064,.064),-.099,.019],dtype=np.float32))
  if max(abs(delta))<.001 and abs(dt)<.001:break
 for k in range(14):s,r,d,tr,i=e.step(np.array([0,0,0,0,-.019],dtype=np.float32))
 print(seed,'base',*[round(s.get(rb,f),3) for f in ['x','y','theta']],'hook',*[round(s.get(h,f),3) for f in ['x','y','theta','held']],flush=True)
 for k in range(10):s,r,d,tr,i=e.step(np.array([0,.049,0,0,0],dtype=np.float32))
 print('MOVE',*[round(s.get(h,f),3) for f in ['x','y','theta','held']],flush=True)
 e.close()
