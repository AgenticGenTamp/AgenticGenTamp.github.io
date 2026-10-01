import numpy as np, kutil
from env_client import make_env
kutil.MZ=0.269
env=make_env(); o,info=env.reset(seed=0)
c=kutil.Ctl(env,o)
cu=c.opos("cube1")
c.gotobase(-0.15,-0.24,3.13)
b,q,g=c.robot(); yaw=b[2]
print("base",np.round(b,3))
c.move_to([cu[0],cu[1],0.4],yaw=yaw)
succ=[]
for dz in [0.0,0.02]:
  for dx in [-0.03,-0.015,0.0,0.015,0.03]:
    for dy in [-0.03,-0.015,0.0,0.015,0.03]:
        p=[cu[0]+dx,cu[1]+dy,cu[2]+dz]
        ok=c.move_to(p,yaw=yaw)
        c.grip(True); gg=c.robot()[2]
        if gg>0.5:
            succ.append((dx,dy,dz))
            # release: move back down to rest
            c.move_to([cu[0]+dx,cu[1]+dy,cu[2]],yaw=yaw); c.grip(False)
            if c.robot()[2]>0.5: print("stuck!"); break
print("successes",succ, "steps",c.steps)
env.close()
