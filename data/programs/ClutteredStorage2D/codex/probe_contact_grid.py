import math
import numpy as np
from env_client import make_env


def move(env, s, gx, gy, gt, ga, vac, steps=100):
    r=s.get_object_from_name("robot")
    for _ in range(steps):
        rx,ry,th,arm=(float(s.get(r,q)) for q in ("x","y","theta","arm_joint"))
        er=(gt-th+math.pi)%(2*math.pi)-math.pi
        a=np.array([max(-.05,min(.05,gx-rx)),max(-.05,min(.05,gy-ry)),max(-.196,min(.196,er)),max(-.1,min(.1,ga-arm)),vac],np.float32)
        s,*_=env.step(a)
        if max(abs(gx-float(s.get(r,"x"))),abs(gy-float(s.get(r,"y"))),abs(er),abs(ga-float(s.get(r,"arm_joint"))))<.002: break
    return s

for axis in ("x","y"):
  for sign in (-1,1):
    for off in (.2,.3,.4,.5,.6,.7):
      env=make_env(); s,_=env.reset(seed=1); b=s.get_object_from_name("block1")
      bx,by=(float(s.get(b,q)) for q in ("x","y")); gx,gy=bx,by
      if axis=="x": gx+=sign*off
      else: gy+=sign*off
      s=move(env,s,gx,gy,0.,.2,0)
      old=(float(s.get(b,"x")),float(s.get(b,"y")))
      # Turn on, then move diagonally to expose attachment robustly.
      s=move(env,s,gx+.08,gy+.06,0.,.2,1,steps=4)
      new=(float(s.get(b,"x")),float(s.get(b,"y")))
      delta=math.hypot(new[0]-old[0],new[1]-old[1])
      print(axis,sign,off,"delta",round(delta,4),"old/new",tuple(round(z,3) for z in old+new))
      env.close()
