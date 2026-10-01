from env_client import make_env
import numpy as np
env=make_env()
J=[f'joint_{i}' for i in range(1,8)]
for j in range(7):
  for s in [1,-1]:
    obs,_=env.reset(seed=0); r=obs.get_object_from_name('robot')
    prev=None
    for k in range(40):
        a=np.zeros(11,dtype=np.float32); a[3+j]=0.2*s
        obs,*_=env.step(a); v=obs.get(r,J[j])
        if prev is not None and abs(v-prev)<1e-9: break
        prev=v
    print(J[j], s, round(v,3), 'steps',k)
env.close()
