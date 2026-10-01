import numpy as np
from env_client import make_env
np.set_printoptions(precision=4, suppress=True, linewidth=150)
env = make_env()
obs, info = env.reset(seed=0)
def pr(o): print(o[125:136], "vel", o[136:147])
pr(obs)
for t in range(5):
    a=np.zeros(11,dtype=np.float32); a[3]=0.1; a[0]=0.1
    obs,r,te,tr,info=env.step(a); pr(obs); print(r)
for t in range(3):
    a=np.zeros(11,dtype=np.float32)
    obs,r,te,tr,info=env.step(a); pr(obs)
for t in range(5):
    a=np.zeros(11,dtype=np.float32); a[10]=1.0
    obs,r,te,tr,info=env.step(a); pr(obs)
for t in range(3):
    a=np.zeros(11,dtype=np.float32); a[2]=0.1
    obs,r,te,tr,info=env.step(a); pr(obs)
env.close()
