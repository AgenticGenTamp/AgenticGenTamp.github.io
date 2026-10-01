import numpy as np
from env_client import make_env
env=make_env(); obs,_=env.reset(seed=0); o0=np.asarray(obs,float)
a=np.zeros(11,dtype=np.float32); a[3]=0.1  # joint1
for t in range(10):
    obs,r,te,tr,_=env.step(a)
o=np.asarray(obs,float)
print("joint deltas after 10 steps of +0.1 on arm j1:",np.round(o[19:26]-o0[19:26],4))
a[:]=0; a[0]=0.1
obs,_=env.reset(seed=0); o0=np.asarray(obs,float)
for t in range(10): obs,r,te,tr,_=env.step(a)
print("base delta after 10 steps +0.1 x:",np.round(np.asarray(obs,float)[16:19]-o0[16:19],4))
a[:]=0; a[10]=1.0
obs,_=env.reset(seed=0)
for t in range(10): obs,r,te,tr,_=env.step(a)
print("gripper after 10 steps cmd1:",np.asarray(obs,float)[26])
a[10]=0.0
for t in range(10): obs,r,te,tr,_=env.step(a)
print("gripper after 10 steps cmd0:",np.asarray(obs,float)[26])
