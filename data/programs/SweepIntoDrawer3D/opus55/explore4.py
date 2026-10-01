import numpy as np
from env_client import make_env
np.set_printoptions(precision=4, suppress=True, linewidth=150)
env = make_env()
obs, info = env.reset(seed=0)
q0=obs[128:135].copy()
qd=q0+np.array([0.6,0.4,0.3,-0.5,0.2,0.5,-0.3])
bd=obs[125:128]+np.array([0.3,-0.2,0.3])
for t in range(40):
    a=np.zeros(11,dtype=np.float32)
    a[3:10]=np.clip(qd-obs[128:135],-0.1,0.1)
    a[0:3]=np.clip(bd-obs[125:128],-0.1,0.1)
    obs,*_=env.step(a)
    if t%3==0 or t>30: print(t, obs[128:135]-qd, obs[125:128]-bd)
env.close()
