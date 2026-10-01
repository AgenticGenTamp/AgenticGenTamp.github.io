import math
import numpy as np
from env_client import make_env

def f(s,o,k):return float(s.get(o,k))
def wrap(a):return (a+math.pi)%(2*math.pi)-math.pi
env=make_env();s,inf=env.reset(seed=450);sp=env.observation_space;r=s.get_objects(sp.get_type("kin_robot"))[0];h=s.get_objects(sp.get_type("hook"))[0];hx,hy=f(s,h,"x"),f(s,h,"y")
print("hook",hx,hy)
def act(tx,ty,tt,dg,n):
 global s
 for _ in range(n):
  a=np.array([np.clip(tx-f(s,r,"x"),-.03,.03),np.clip(ty-f(s,r,"y"),-.03,.03),np.clip(wrap(tt-f(s,r,"theta")),-.098,.098),0,dg],np.float32);s,*_=env.step(a)
# route right
act(3.0,2.1,-math.pi/2,.015,80)
for y in [.75,.85,.95,1.05,1.15,1.25]:
 for xoff in [-.1,-.05,0,.05,.1]:
  act(hx+xoff,1.5,-math.pi/2,.015,35);act(hx+xoff,y,-math.pi/2,.015,35);act(hx+xoff,y,-math.pi/2,-.015,16)
  print("try",xoff,y,"actual",round(f(s,r,"x"),2),round(f(s,r,"y"),2),"held",f(s,h,"held"),"gap",round(f(s,r,"finger_gap"),2))
  if f(s,h,"held")>.5:break
 if f(s,h,"held")>.5:break
env.close()
