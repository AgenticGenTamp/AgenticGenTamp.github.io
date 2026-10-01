import numpy as np
from env_client import make_env
from experiment_grasp import action_pose

def trial(d):
 e=make_env();s,_=e.reset(seed=0);h0=s[9:12].copy()
 u=np.array([np.cos(s[11]),np.sin(s[11])]);n=np.array([u[1],-u[0]])
 q=s[9:11]-s[18]*u+.30*u; base=q+d*n; stage=base+.25*n; heading=np.arctan2(-n[1],-n[0])
 poses=[(np.array([stage[0],.16]),.1,0,50),(stage,.1,0,50),(base,.1,0,20),(base,.2,1,10),(base+.15*n,.2,1,15)]
 for pos,arm,vac,N in poses:
  for _ in range(N): s,*_=e.step(action_pose(s,pos=pos,theta=heading,arm=arm,vac=vac))
 e.close();return s,h0,base

for d in np.arange(.15,.51,.025):
 s,h,b=trial(d); print(round(d,3),'hookmove',round(float(np.linalg.norm(s[9:11]-h[:2])),3),'hth',round(float(s[11]-h[2]),3),'robot',np.round(s[[0,1,4,6]],3),'base',np.round(b,3))
