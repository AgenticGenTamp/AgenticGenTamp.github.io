import math
import numpy as np
from env_client import make_env

def f(s,o,k): return float(s.get(o,k))
def wrap(a): return (a+math.pi)%(2*math.pi)-math.pi
def run(seed, ox, oy, theta):
 env=make_env();s,inf=env.reset(seed=seed);sp=env.observation_space;r=s.get_objects(sp.get_type("kin_robot"))[0];h=s.get_objects(sp.get_type("hook"))[0]
 hx,hy=f(s,h,"x"),f(s,h,"y"); target=(hx+ox,hy+oy)
 print("START",seed,"hook",round(hx,2),round(hy,2),"robot",round(f(s,r,"x"),2),round(f(s,r,"y"),2),round(f(s,r,"theta"),2),"target",target,theta)
 phases=[(max(2.3,target[0]),2.1,theta,0.015,70),(target[0],2.1,theta,.015,50),(target[0],target[1],theta,.015,70),(target[0],target[1],theta,-.015,30)]
 for tx,ty,tt,dg,n in phases:
  for _ in range(n):
   a=np.array([np.clip(tx-f(s,r,"x"),-.03,.03),np.clip(ty-f(s,r,"y"),-.03,.03),np.clip(wrap(tt-f(s,r,"theta")),-.098,.098),-.08,dg],np.float32)
   s,re,term,tr,info=env.step(a)
  print(" ph",round(f(s,r,"x"),2),round(f(s,r,"y"),2),round(f(s,r,"theta"),2),"gap",round(f(s,r,"finger_gap"),2),"held",f(s,h,"held"),"hookpos",round(f(s,h,"x"),2),round(f(s,h,"y"),2))
 env.close()

tests=[(0,.9,-math.pi/2),(0,.75,-math.pi/2),(0,.9,math.pi/2),(-.4,.45,0),(.4,.45,math.pi)]
for i,t in enumerate(tests):run(400+i,*t)
