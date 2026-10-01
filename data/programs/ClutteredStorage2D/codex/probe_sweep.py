import math
import numpy as np
from env_client import make_env

env = make_env(); s,_=env.reset(seed=1)
r=s.get_object_from_name("robot"); b=s.get_object_from_name("block1")
bx,by=(float(s.get(b,q)) for q in ("x","y"))
phase=0
for i in range(150):
    rx,ry,th,arm=(float(s.get(r,q)) for q in ("x","y","theta","arm_joint"))
    if i < 40:
        gx,gy=bx-1.0,by; gt=0.; ga=.2; vac=0
    else:
        gx,gy=bx+1.0,by; gt=0.; ga=.2; vac=1
    err=(gt-th+math.pi)%(2*math.pi)-math.pi
    a=np.array([max(-.05,min(.05,gx-rx)),max(-.05,min(.05,gy-ry)),max(-.196,min(.196,err)),max(-.1,min(.1,ga-arm)),vac],np.float32)
    s,rew,done,trunc,_=env.step(a)
    attached=float(s.get(r,"vacuum"))
    if attached or i%5==0:
        print(i,"base",round(float(s.get(r,"x")),3),round(float(s.get(r,"y")),3),"th",round(float(s.get(r,"theta")),3),"arm",round(float(s.get(r,"arm_joint")),3),"vac",attached,"block",round(float(s.get(b,"x")),3),round(float(s.get(b,"y")),3))
    if attached and i>80: break
env.close()
