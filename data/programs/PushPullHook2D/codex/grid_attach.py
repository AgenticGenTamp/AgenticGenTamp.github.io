import numpy as np
from env_client import make_env
from experiment_grasp import action_pose

def trial(d,hoff,arm):
 e=make_env();s,_=e.reset(seed=0);h0=s[9:12].copy()
 u=np.array([np.cos(s[11]),np.sin(s[11])]);n=np.array([u[1],-u[0]])
 q=s[9:11]-s[18]*u+.25*u
 # robot lies +n from line; requested heading is toward line plus offset
 base=q+d*n; stage=base+.25*n; heading=np.arctan2(-n[1],-n[0])+hoff
 for pos,ar,vac,N in [(np.array([stage[0],.16]),.1,0,35),(stage,.1,0,30),(base,.1,0,10),(base,arm,0,3),(base,arm,1,3),(base+.12*n,arm,1,5)]:
  for _ in range(N):s,*_=e.step(action_pose(s,pos=pos,theta=heading,arm=ar,vac=vac))
 move=np.linalg.norm(s[9:11]-h0[:2]);e.close();return move

for hoff in [0,np.pi/2,-np.pi/2,np.pi]:
 for arm in [.1,.2]:
  hits=[]
  for d in np.arange(.18,.711,.01):
   if trial(float(d),hoff,arm)>.005:hits.append(round(float(d),2))
  print('headingoff',round(hoff,2),'arm',arm,'hits',hits)
