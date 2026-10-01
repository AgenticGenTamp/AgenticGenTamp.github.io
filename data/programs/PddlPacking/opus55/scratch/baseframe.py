from env_client import make_env
import numpy as np
env=make_env(); obs,_=env.reset(seed=0)
T=env.observation_space.get_type; r=obs.get_objects(T("robot"))[0]
for k in range(8):
    a=np.zeros(11,dtype=np.float32); a[2]=0.2; obs,*_=env.step(a)
a=np.zeros(11,dtype=np.float32); a[0]=-0.2; obs,*_=env.step(a)
print([round(float(obs.get(r,f)),4) for f in ["base_x","base_y","base_rot"]])
