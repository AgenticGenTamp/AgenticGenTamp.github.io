from env_client import make_env
import numpy as np
np.set_printoptions(precision=3, suppress=True)
env = make_env()
obs,_ = env.reset(seed=1)
print(obs[0:3], obs[16:18])
# go to x below block by 0.8 in world y
tgt = obs[0:2] + np.array([0.0,-0.9])
def goto(obs,tgt,n=200):
    for i in range(n):
        d = tgt-obs[16:18]
        if np.linalg.norm(d)<1e-3: break
        a = np.clip(d,-0.0499,0.0499)
        obs,r,te,tr,info=env.step(a)
    return obs
obs=goto(obs,tgt)
print('at', obs[16:18], obs[0:6])
for i in range(30):
    obs,r,te,tr,info=env.step(np.array([0,0.0499]))
    print(i, obs[16:18], obs[0:6])
for i in range(15):
    obs,r,te,tr,info=env.step(np.array([0,-0.0499]))
    print('back',i, obs[16:18], obs[0:6])
