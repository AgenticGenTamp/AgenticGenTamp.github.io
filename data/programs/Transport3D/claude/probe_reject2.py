import numpy as np
from probe_util import *
from probe_arm import uparm, dr
env=make_env(); obs,_=env.reset(seed=3); obs=uparm(env,obs); obs=dr(env,obs,0.232,0.0)
b=rs(obs); print("at table",fmt(b[:3]))
o2,r,t,tr,i=env.step(act(bx=0.2,j1=0.2))       # base blocked + valid joint
print("blockedbase+validj1 delta",fmt(rs(o2)-b),"r",r)
b=rs(o2)
o2,r,t,tr,i=env.step(act(bx=0.2,by=0.2,j1=0.2))  # blocked x, free y
print("blockedx+freey+j1 delta",fmt(rs(o2)-b))
b=rs(o2)
o2,r,t,tr,i=env.step(act(bx=0.05,j1=0.2))
print("small blocked bx=0.05 + j1 delta",fmt(rs(o2)-b))
b=rs(o2)
o2,r,t,tr,i=env.step(act(bx=0.002,j1=0.2))
print("tiny bx=0.002 + j1 delta",fmt(rs(o2)-b))
env.close()
