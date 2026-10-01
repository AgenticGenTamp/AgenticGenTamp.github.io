"""Measure whether repeated low-speed re-engagement moves the can upright."""
import numpy as np
from env_client import make_env

env=make_env(); s,_=env.reset(seed=0); initial=s.copy()
home=s[96:103].copy(); pose=home.copy(); pose[1]=.67; pose[3]=-1.43

def step(a):
    global s
    s,r,t,tr,i=env.step(np.asarray(a,np.float32)); return r,t

def joints(target, limit=60):
    for _ in range(limit):
        a=np.zeros(11); err=target-s[96:103]; a[3:10]=np.clip(err*.5,-.1,.1); step(a)
        if np.max(abs(err))<.04: break

def yservo(target, limit=30):
    for _ in range(limit):
        a=np.zeros(11); a[1]=np.clip((target-s[94])*.5,-.1,.1); step(a)
        if abs(target-s[94])<.015: break

for cycle in range(3):
    joints(home); yservo(float(s[33]-.28)); joints(pose)
    oldy=float(s[33]); stalled=0
    for k in range(80):
        a=np.zeros(11); a[1]=.01; step(a)
        dy=float(s[33]-oldy)
        stalled = stalled+1 if abs(dy)<2e-4 else 0
        oldy=float(s[33])
        if stalled>12: break
    print(cycle,'can',np.round(s[32:39],4).tolist(),'basey',round(float(s[94]),3),'steps',k)
env.close()
