import numpy as np
from env_client import make_env
env=make_env()
for seed in range(4):
    obs,info=env.reset(seed=seed)
    out=[]
    for o in obs:
        if o.type.name=="block" and o.name.startswith("green") and o.name!="green0":
            d=obs.data[o]; qx,qy,qz,qw=d[3:7]
            yaw=np.degrees(np.arctan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz)))
            out.append((o.name,round(float(d[0]),3),round(float(d[1]),3),round(float(yaw),1)))
    print(seed,out)
env.close()
