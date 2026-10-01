from env_client import make_env
from approach import GeneratedApproach
import numpy as np
import sys
seed=0;off=float(sys.argv[1]);h=float(sys.argv[2]);E=make_env();s,i=E.reset(seed=seed)
p=GeneratedApproach(E.action_space,E.observation_space,{});p.reset(s,i)
for k,(pt,g) in enumerate(p.waypoints):
 if k<4:pt[0]+=off
 if k in [1,2]:pt[2]=h
for k in range(200):
 s,r,t,tr,i=E.step(p.get_action(s))
 if k%20==0:
  cs=np.array([[s.get(o,f) for f in ['x','y','z']] for o in p.cubes]);print(k,p.stage,r,'maxz',round(cs[:,2].max(),4),'maxy',round(cs[:,1].max(),4),flush=True)
 if t or tr:break
print('END',off,h,r,[(o.name,[round(s.get(o,f),4) for f in ['x','y','z']]) for o in p.cubes if s.get(o,'y')>0 or s.get(o,'z')>.5],flush=True)
E.close()
