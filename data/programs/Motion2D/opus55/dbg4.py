from env_client import make_env
import numpy as np
env = make_env()
obs, info = env.reset(seed=159)  # wall x=0.5 spans y 0..2.3
r=obs.get_object_from_name('robot')
def st(a):
    global obs
    obs,*_=env.step(np.array(a,dtype=np.float32))
for th_goal in [0.0, np.pi/2]:
    obs, info = env.reset(seed=159)
    for k in range(20):
        th=float(obs.get(r,'theta')); d=np.clip(np.arctan2(np.sin(th_goal-th),np.cos(th_goal-th)),-0.19,0.19); st([0,0,d,0,0])
    for d in [0.05,0.01,0.002,0.0005,0.0001]:
        for i in range(10): st([d,0,0,0,0])
    print(th_goal, float(obs.get(r,'theta')), 'max x', float(obs.get(r,'x')), 'arm', float(obs.get(r,'arm_joint')))
