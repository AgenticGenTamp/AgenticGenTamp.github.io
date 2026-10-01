import numpy as np
from env_client import make_env
from experiment_grasp import action_pose
def tr(d):
 e=make_env();s,_=e.reset(seed=2);h=s[9:11].copy();u=np.array([np.cos(s[11]),np.sin(s[11])]);n=-np.array([u[1],-u[0]]);q=s[9:11]-s[18]*u+.3*u;base=q+d*n;stage=base+.2*n;th=np.arctan2(-n[1],-n[0])
 for pos,ar,v,N in [(np.array([stage[0],.16]),.1,0,25),(stage,.1,0,20),(base,.1,0,8),(base,.2,0,2),(base,.2,1,3),(base+.12*n,.2,1,4)]:
  for _ in range(N):s,*_=e.step(action_pose(s,pos=pos,theta=th,arm=ar,vac=v))
 m=np.linalg.norm(s[9:11]-h);e.close();return m
for d in np.arange(.12,.501,.005):
 m=tr(float(d))
 if m>.003:print(round(float(d),3),round(float(m),3))
