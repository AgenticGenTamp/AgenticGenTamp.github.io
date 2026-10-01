from env_client import make_env
import numpy as np
env = make_env()
obs,_=env.reset(seed=0); v0=np.array(obs.vec(obs.get_objects() if not callable(getattr(obs,"get_objects")) else list(obs.data.keys())))
for g in [-1,-1,-1,1,1]:
    a=np.zeros(11,np.float32); a[10]=g; obs,*_=env.step(a)
    v=np.array(obs.vec(obs.get_objects() if not callable(getattr(obs,"get_objects")) else list(obs.data.keys()))); print(g, "changed idx", np.where(np.abs(v-v0)>1e-9)[0])
