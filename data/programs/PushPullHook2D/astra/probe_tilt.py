from env_client import make_env
from approach import GeneratedApproach,wrap
import numpy as np
import math
for seed in [0,1,2,3,6]:
 e=make_env();s,_=e.reset(seed=seed);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,{})
 for k in range(500):
  a=p.get_action(s);s,r,t,tr,_=e.step(a)
  if p.phase==6:break
 print('seed',seed,'startphase',p.phase,'state',s[:12], 'buttons',s[[20,21,29,30]])
 if p.phase!=6:e.close();continue
 # Move away from button along vertical before tilt; then use rotation for crossbar normal toward target
 d=s[29:31]-s[20:22];side=1 if d[1]>=0 else -1
 desired=math.atan2(d[1]*side,d[0]*side)
 for k in range(20):
  da=wrap(desired-s[11]);s,*_=e.step(np.array([0,0,np.clip(da,-.05,.05),0,1],dtype=np.float32))
  if abs(da)<.001:break
 u=np.array([np.cos(s[11]),np.sin(s[11])]);v=np.array([-u[1],u[0]])
 # short arm center is hook corner -0.3v, push toward button along u*side
 target=s[20:22]+.3*v-.12*side*u
 for k in range(150):
  er=target-s[9:11];s,*_=e.step(np.array([*np.clip(er,-.025,.025),0,0,1],dtype=np.float32))
  if np.linalg.norm(er)<.005:break
 for k in range(60):
  old=s.copy();s,r,t,tr,_=e.step(np.array([*(.01*side*u),0,0,1],dtype=np.float32))
  db=s[20:22]-old[20:22]
  if np.linalg.norm(db)>.001 or k%10==0:print('push',k,'u',u.round(3),'hook',s[9:11].round(3),'buttondelta',db.round(4),'goalerr',(s[29:31]-s[20:22]).round(3),'done',t)
  if t:break
 e.close()
