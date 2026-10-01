import numpy as np
from env_client import make_env

e=make_env(); s,_=e.reset(seed=11); typ=e.observation_space.get_type
r=s.get_objects(typ('kin_robot'))[0]
for _ in range(20):
 th=float(s.get(r,'theta')); s,*_=e.step(np.array([0,.03,np.clip(-th,-.19,.19),0,0],float))
for extension in (0,.099):
 if extension: 
  for _ in range(3): s,*_=e.step(np.array([0,0,0,.099,0],float))
 s,*_=e.step(np.array([0,0,.01,0,0],float))
 print('joint',s.get(r,'arm_joint'))
 for part in ('base','arm','gripper_l','gripper_r'):
  om=float(s.get(r,'omega_'+part)); vx=float(s.get(r,'vx_'+part)); vy=float(s.get(r,'vy_'+part))
  print(part,'v',vx,vy,'om',om,'relative inferred',(-vy/om if om else 0,vx/om if om else 0))
e.close()
