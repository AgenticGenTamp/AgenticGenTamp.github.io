import numpy as np
from env_client import make_env
from experiment_grasp import action_pose

def trial(seed, along, dist, arm, side=-1):
 e=make_env();s,_=e.reset(seed=seed); init=s.copy()
 u=np.array([np.cos(s[11]),np.sin(s[11])]); n=side*np.array([u[1],-u[0]])
 tip=s[9:11]-s[18]*u; q=tip+along*u; pos=q-dist*n; heading=np.arctan2(n[1],n[0])
 for t in range(120):
  if t<10:a=action_pose(s,arm=.1)
  elif t<35:a=action_pose(s,theta=heading,arm=.1)
  elif t<90:a=action_pose(s,pos=pos,theta=heading,arm=.1)
  else:a=action_pose(s,pos=pos,theta=heading,arm=arm,vac=1)
  s,*_=e.step(a)
 moved=np.linalg.norm(s[9:11]-init[9:11]);e.close()
 return moved,s, pos, heading

if __name__=='__main__':
 for along in [.05,.15,.3,.5]:
  for dist in [.12,.16,.2,.24,.28,.32,.36,.4]:
   for side in [-1,1]:
    moved,s,p,h=trial(0,along,dist,.2,side)
    if moved>.0001 or s[6]>.5: print('HIT',along,dist,side,moved,np.round(s[9:12],3),'vac',s[6],p,h)
