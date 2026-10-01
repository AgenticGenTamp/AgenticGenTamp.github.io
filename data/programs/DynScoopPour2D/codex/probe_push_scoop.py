import math
import numpy as np
from env_client import make_env

def f(s,o,k):return float(s.get(o,k))
def wrap(a):return (a+math.pi)%(2*math.pi)-math.pi
def run(seed):
 env=make_env();s,inf=env.reset(seed=seed);sp=env.observation_space;r=s.get_objects(sp.get_type("kin_robot"))[0];h=s.get_objects(sp.get_type("hook"))[0];sm=[s.get_object_from_name(n) for n in s.get_object_names() if n.startswith("small")];hx=f(s,h,"x")
 def phase(tx,ty,tt=-math.pi/2,dg=0,n=60):
  nonlocal s
  term=False
  for _ in range(n):
   a=np.array([np.clip(tx-f(s,r,"x"),-.03,.03),np.clip(ty-f(s,r,"y"),-.03,.03),np.clip(wrap(tt-f(s,r,"theta")),-.098,.098),0,dg],np.float32);s,re,term,tr,info=env.step(a)
   if term or tr:break
  vals=[(f(s,o,"x"),f(s,o,"y")) for o in sm]
  print("p",tx,ty,round(tt,2),"R",sum(x>1.75 for x,y in vals),"hi",sum(y>1.4 for x,y in vals),"max",tuple(round(q,2) for q in max(vals,key=lambda z:z[0])),"rob",round(f(s,r,"x"),2),round(f(s,r,"y"),2),"hook",round(f(s,h,"x"),2),round(f(s,h,"y"),2),"held",f(s,h,"held"),"term",term)
  return term
 phase(3.0,2.1,dg=.015,n=80);phase(hx-.1,1.5,dg=.015,n=40);phase(hx-.1,.75,dg=.015,n=35);phase(hx-.1,.75,dg=-.015,n=20)
 # clear wall carrying hook, place corner at left floor, sweep into pile, scoop upward, cross and tip
 for args in [(hx-.1,2.4,-math.pi/2,0,65),(.35,2.4,-math.pi/2,0,110),(.35,.75,-math.pi/2,0,80),(1.25,.75,-math.pi/2,0,50),(1.25,2.4,-math.pi/2,0,65),(2.6,2.4,-math.pi/2,0,55),(2.6,2.2,0,0,40),(2.6,2.2,0,.015,20)]:
  if phase(*args):break
 env.close()
for q in [502,503]:run(q)
