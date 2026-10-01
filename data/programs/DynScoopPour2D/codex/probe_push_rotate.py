import math
import numpy as np
from env_client import make_env
def f(s,o,k):return float(s.get(o,k))
def wrap(a):return (a+math.pi)%(2*math.pi)-math.pi
env=make_env();s,inf=env.reset(seed=560);sp=env.observation_space;r=s.get_objects(sp.get_type("kin_robot"))[0];h=s.get_objects(sp.get_type("hook"))[0];sm=[s.get_object_from_name(n) for n in s.get_object_names() if n.startswith("small")];hx=f(s,h,"x"); target_theta=-math.pi/2
def go(tx,ty,n,dg=0,speed=.03,tt=None):
 global s,target_theta
 if tt is not None:target_theta=tt
 for _ in range(n):
  a=np.array([np.clip(tx-f(s,r,"x"),-speed,speed),np.clip(ty-f(s,r,"y"),-speed,speed),np.clip(wrap(target_theta-f(s,r,"theta")),-.098,.098),0,dg],np.float32);s,re,term,tr,info=env.step(a)
  if term:break
 vals=[(f(s,o,"x"),f(s,o,"y")) for o in sm];print("go",tx,ty,round(target_theta,2),"robot",round(f(s,r,"x"),2),round(f(s,r,"y"),2),round(f(s,r,"theta"),2),"R",sum(x>1.75 for x,y in vals),"H",sum(y>1.4 for x,y in vals),"term",term,"hook",round(f(s,h,"x"),2),round(f(s,h,"y"),2))
 return term
go(3,2.1,80,.015);go(hx-.1,.75,70,.015);go(hx-.1,.75,20,-.015);go(hx-.1,2.4,65);go(.25,2.4,110);go(.25,.75,70);go(1.3,.75,80,speed=.015)
# rotate up at wall, then translate right/release
for tt in [-1.2,-.8,-.4,0]:
 if go(1.3,.75,12,tt=tt):break
go(2.6,.75,60);go(2.6,.75,20,dg=.015)
env.close()
