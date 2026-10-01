from env_client import make_env
import numpy as np
np.set_printoptions(precision=4,suppress=True)
for arm in [.1,.2]:
 for theta in [np.pi,0,np.pi/2,-np.pi/2]:
  e=make_env();s,i=e.reset(seed=0)
  u=np.array([np.cos(s[11]),np.sin(s[11])]);p=s[9:11]-1.22*u
  direction=np.array([np.cos(theta),np.sin(theta)])
  for j in range(100):
   th=np.arctan2(np.sin(theta-s[2]),np.cos(theta-s[2]));d=p-.4*direction-s[:2]
   a=np.array([*np.clip(d,-.05,.05),np.clip(th,-.196,.196),np.clip(arm-s[4],-.1,.1),0],dtype=np.float32)
   s,*_=e.step(a)
   if np.max(np.abs(d))<1e-5 and abs(th)<1e-5:break
  before=s.copy()
  for j in range(80):
   old=s.copy();s,*_=e.step(np.array([*(.005*direction),0,0,1],dtype=np.float32))
   if np.linalg.norm(s[9:11]-old[9:11])>1e-6:
    print('grasp arm',arm,'theta',theta,'j',j,'beforebase',old[:5], 'hookdelta',s[9:12]-old[9:12]);break
  else: print('NO arm',arm,'theta',theta,'base',s[:5],'hook',s[9:12])
  e.close()
e=make_env();s,i=e.reset(seed=0)
u=np.array([np.cos(s[11]),np.sin(s[11])]);p=s[9:11]-1.22*u
for j in range(100):
 th=np.arctan2(np.sin(np.pi-s[2]),np.cos(np.pi-s[2]));d=p+np.array([.4,0])-s[:2]
 s,*_=e.step(np.array([*np.clip(d,-.05,.05),np.clip(th,-.196,.196),-.1,0],dtype=np.float32))
 if np.max(np.abs(d))<1e-5 and abs(th)<1e-5:break
for j in range(46):s,*_=e.step(np.array([-.005,0,0,0,1],dtype=np.float32))
for a in [[.05,0,0,0,1],[0,.05,0,0,1],[0,0,.1,0,1],[0,0,0,.1,1],[0,0,0,0,0],[.05,0,0,0,0]]:
 old=s.copy();s,*_=e.step(np.array(a,dtype=np.float32));print('held action',a,'robotchange',s[:5]-old[:5],'hookdelta',s[9:12]-old[9:12])
e.close()
