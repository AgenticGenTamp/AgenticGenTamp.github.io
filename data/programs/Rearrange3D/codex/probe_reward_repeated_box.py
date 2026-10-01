"""Repeated upright box nudges with seed 11's can already beside the bowl."""
import numpy as np
from env_client import make_env

env=make_env(); s,_=env.reset(seed=11); home=s[96:103].copy()
pose=home.copy(); pose[1]=.67
reach=float(s[16]-s[93]); pose[3]=float(np.clip(-1.70+3.2*(reach-.542),-1.95,-1.40))
last_r=-1.0

def step(a):
 global s,last_r
 s,r,t,tr,_=env.step(np.asarray(a,np.float32))
 if r != last_r or t:
  print('REWARD',r,t,'xyz',np.round(s[[0,1,16,17,18,32,33,34]],4).tolist(),flush=True)
  last_r=r
 return r,t

def servo_arm(goal):
 for _ in range(60):
  e=goal-s[96:103]
  if np.max(np.abs(e))<.045: break
  a=np.zeros(11);a[3:10]=np.clip(e*.5,-.1,.1);a[10]=1;step(a)

def servo_y(goal):
 for _ in range(15):
  e=goal-float(s[94])
  if abs(e)<.025:break
  a=np.zeros(11);a[1]=np.clip(e*.5,-.1,.1);a[10]=1;step(a)

print('start',np.round(s[[0,1,16,17,18,32,33,34]],4).tolist(),'pose',np.round(pose,2).tolist())
for cycle in range(5):
 servo_arm(home); servo_y(float(s[17])+.28); servo_arm(pose)
 before=s.copy()
 for _ in range(8):
  a=np.zeros(11);a[1]=-.07;a[10]=1
  r,t=step(a)
  if t:break
 print('cycle',cycle,'r',r,'box',np.round(s[16:23],4).tolist(),
       'dist',round(float(np.linalg.norm(s[16:18]-s[0:2])),4),flush=True)
 if t:break
print('final',r,t,np.round(s[[0,1,16,17,18,32,33,34]],4).tolist())
env.close()
