import numpy as np
from env_client import make_env

def f(s,o,k): return float(s.get(o,k))
def run(seed,xcol):
 env=make_env();s,inf=env.reset(seed=seed);r=s.get_objects(env.observation_space.get_type("kin_robot"))[0]; sm=[s.get_object_from_name(n) for n in s.get_object_names() if n.startswith("small")]
 term=False; peakr=0; peakh=0
 # route over top and descend
 for tx,ty,n in [(xcol,2.2,60),(xcol,.27,80)]:
  for _ in range(n):
   a=np.array([np.clip(tx-f(s,r,"x"),-.03,.03),np.clip(ty-f(s,r,"y"),-.03,.03),0,-.08,0],np.float32);s,re,term,tr,info=env.step(a)
 # accelerate straight upward then up-right without braking
 for i in range(90):
  if i<50: a=[0,.03,0,-.08,0]
  else: a=[.03,.03,0,-.08,0]
  s,re,term,tr,info=env.step(np.array(a,np.float32))
  vals=[(f(s,o,"x"),f(s,o,"y")) for o in sm]; peakr=max(peakr,sum(x>1.75 for x,y in vals)); peakh=max(peakh,sum(y>1.5 for x,y in vals))
  if i%10==0 or term: print(seed,i,"rob",round(f(s,r,"x"),2),round(f(s,r,"y"),2),"R",sum(x>1.75 for x,y in vals),"H",sum(y>1.5 for x,y in vals),"ymax",round(max(y for x,y in vals),2),"term",term)
  if term or tr: break
 print("RESULT",seed,"n",len(sm),"peakR",peakr,"peakH",peakh,"term",term)
 env.close()
for seed,x in [(300,1.2),(301,1.35),(302,1.45)]:run(seed,x)
