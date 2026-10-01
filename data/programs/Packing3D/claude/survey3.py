import numpy as np
from env_client import make_env
import pl
env=make_env(); d=pl.Drv(env)
seen=set()
for oc in (1,2,3,4):
  for sd in range(8):
    o=d.reset(seed=sd,oc=oc)
    for n in pl.parts(o):
        f=pl.feat(o,n)
        k=tuple(sorted((a,round(b,4)) for a,b in f.items() if a not in ('pose_x','pose_y','pose_z','pose_qx','pose_qy','pose_qz','pose_qw','grasp_active')))
        if k not in seen:
            seen.add(k); print(oc,sd,n,k,flush=True)
env.close()
