from env_client import make_env
import numpy as np
import sys
for seed in map(int,sys.argv[1:] or range(10)):
 E=make_env();s,_=E.reset(seed=seed);r=s.get_objects(E.observation_space.get_type('kin_robot'))[0];n=0;term=False
 for feature,target,limit,speed in [('y',2.7,25,.03),('theta',0,50,.098),('x',.23,100,.03),('y',.2,100,.03),('x',1.36,80,.03),('y',1.85,120,.015),('x',2.9,60,.03),('theta',1.57,20,.098),('y',1.85,20,.0)]:
  for i in range(limit):
   ix={'x':0,'y':1,'theta':2}[feature];d=target-s.get(r,feature)
   if abs(d)<.004 and speed:break
   a=np.zeros(5);a[ix]=np.clip(d,-speed,speed)
   s,_,term,trunc,_=E.step(a);n+=1
   if term or trunc:break
  if term:break
 pts=[(s.get(o,'x'),s.get(o,'y')) for t in ('small_circle','small_square') for o in s.get_objects(E.observation_space.get_type(t))]
 print(seed,'success',term,'steps',n,'right',sum(x>1.8 for x,y in pts),'total',len(pts),'robot',*[round(s.get(r,f),2) for f in ('x','y','theta')],flush=True)
 E.close()
