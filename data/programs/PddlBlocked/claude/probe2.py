from env_client import make_env
import numpy as np
env = make_env()
lims=[]
for j in range(7):
    for sgn in (1,-1):
        obs,_=env.reset(seed=1)
        r=obs.get_object_from_name("robot")
        last=None
        for k in range(60):
            a=np.zeros(11); a[3+j]=sgn*0.2
            obs,_,_,_,_=env.step(a)
            v=obs.data[r][3+j]
            if last is not None and abs(v-last)<1e-6: break
            last=v
        lims.append((j+1,sgn,round(float(last),4),k))
for l in lims: print(l)
env.close()
