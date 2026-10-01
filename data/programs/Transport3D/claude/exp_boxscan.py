import numpy as np, kutil, itertools
from env_client import make_env
kutil.MZ=0.269
env=make_env(); o,info=env.reset(seed=0)
c=kutil.Ctl(env,o)
bx=c.opos("box0"); hx,hy,hz=0.1,0.15,0.1
ang=np.arctan2(bx[1],bx[0]); bp=bx[:2]-0.55*np.array([np.cos(ang),np.sin(ang)])
print("drive",c.gotobase(bp[0],bp[1],ang))
res=[]
for dz in [0.2,0.15,0.1,0.05]:
  for (dx,dy) in [(0,0),(hx,0),(-hx,0),(0,hy),(0,-hy),(hx,hy)]:
    for yaw in [ang, ang+np.pi/2]:
        p=[bx[0]+dx,bx[1]+dy,dz]
        ok1=c.move_to([p[0],p[1],0.5],yaw=yaw)
        ok=c.move_to(p,yaw=yaw)
        c.grip(True); g=c.robot()[2]
        res.append((dz,dx,dy,round(float(yaw),2),ok,g))
        if g>0.5:
            print("GRASP at",res[-1]); break
        c.grip(False)
  else:
    continue
  break
for r in res: print(r)
print("steps",c.steps)
env.close()
