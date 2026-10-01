import math
import numpy as np
from env_client import make_env
def f(s,o,k):return float(s.get(o,k))
def wrap(a):return (a+math.pi)%(2*math.pi)-math.pi
env=make_env();s,inf=env.reset(seed=580);sp=env.observation_space;r=s.get_objects(sp.get_type("kin_robot"))[0];h=s.get_objects(sp.get_type("hook"))[0];sm=[s.get_object_from_name(n) for n in s.get_object_names() if n.startswith("small")];hx=f(s,h,"x")
def drive(tx,ty,n,dg=0,speed=.03):
 global s
 for i in range(n):
  a=np.array([np.clip(tx-f(s,r,"x"),-speed,speed),np.clip(ty-f(s,r,"y"),-speed,speed),np.clip(wrap(-math.pi/2-f(s,r,"theta")),-.098,.098),0,dg],np.float32);s,re,term,tr,info=env.step(a)
  if term:return True
 vals=[(f(s,o,"x"),f(s,o,"y")) for o in sm];print("at",tx,ty,"R",sum(x>1.75 for x,y in vals),"high",sum(y>1.4 for x,y in vals),"coords",[(round(x,2),round(y,2)) for x,y in vals],"hook",round(f(s,h,"x"),2),round(f(s,h,"y"),2));return False
drive(3,2.1,80,.015);drive(hx-.1,.75,70,.015);drive(hx-.1,.75,20,-.015);drive(hx-.1,2.4,65);drive(.25,2.4,110);drive(.25,.75,70);drive(1.5,.75,100,speed=.015);drive(1.5,2.4,70,speed=.03);drive(2.6,2.4,45,speed=.03)
env.close()
