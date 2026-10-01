"""Calibrate q2=.8 hook for live-x stopped can placement."""
import numpy as np
from env_client import make_env

env=make_env()
for seed in (0,3):
 s,_=env.reset(seed=seed); home=s[96:103].copy(); target_x=float(s[0]); target_y=float(s[1]-s[14]-s[46]-.025)
 gy=float(s[33])-.28
 for _ in range(35):
  a=np.zeros(11,np.float32);a[1]=np.clip((gy-s[94])*.5,-.1,.1);s,*_=env.step(a)
  if abs(gy-s[94])<.025:break
 pose=home.copy();pose[1]=.80;pose[3]=-1.43
 for _ in range(75):
  e=pose-s[96:103]
  if np.max(abs(e))<.04:break
  a=np.zeros(11,np.float32);a[3:10]=np.clip(e*.5,-.1,.1);s,*_=env.step(a)
 print('start',seed,'target',round(target_x,3),round(target_y,3),'can',np.round(s[32:35],3).tolist(),flush=True)
 for k in range(80):
  a=np.zeros(11,np.float32);a[1]=.01
  s,r,t,tr,_=env.step(a)
  if k%5==4 or s[32]<=target_x+.01 or t:
   print(seed,k,'r/t',r,t,'can',np.round(s[32:39],4).tolist(),'err',np.round(s[32:34]-[target_x,target_y],4).tolist(),flush=True)
  if s[32]<=target_x+.01 or t:break
 # retreat immediately, retaining pose
 for j in range(20):
  a=np.zeros(11,np.float32);a[0]=-.1;a[3:10]=np.clip((pose-s[96:103])*.5,-.1,.1)
  s,r,t,tr,_=env.step(a)
  if t or tr:break
 print('end',seed,r,t,np.round(s[32:39],4).tolist(),flush=True)
env.close()
