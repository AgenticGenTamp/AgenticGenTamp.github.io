import numpy as np
from env_client import make_env

def trial(seed, sep):
 e=make_env(); s,_=e.reset(seed=seed); typ=e.observation_space.get_type
 r=s.get_objects(typ('kin_robot'))[0]; b=s.get_objects(typ('target_block'))[0]
 bx=float(s.get(b,'x')); by=float(s.get(b,'y'))
 for i in range(60):
  x=float(s.get(r,'x')); y=float(s.get(r,'y')); th=float(s.get(r,'theta'))
  a=np.array([np.clip(bx-x,-.049,.049),np.clip(by+sep-y,-.049,.049),np.clip(-1.5708-th,-.19,.19),0,0],float)
  s,*_=e.step(a)
 for j in range(14): s,*_=e.step(np.array([0,0,0,0,-.019],float))
 y0=float(s.get(b,'y'))
 for j in range(10): s,*_=e.step(np.array([0,.03,0,0,0],float))
 print(seed,sep,'held',float(s.get(b,'held')),'lift',round(float(s.get(b,'y'))-y0,3),
       'btheta',round(float(s.get(b,'theta')),2),flush=True)
 e.close()

for z in (.4,.45,.5,.55,.6,.65): trial(11,z)
