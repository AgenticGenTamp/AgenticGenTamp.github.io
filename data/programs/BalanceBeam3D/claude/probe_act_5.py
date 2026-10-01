import numpy as np
from env_client import make_env
np.set_printoptions(precision=5, suppress=True)

# arm magnitude scaling + clipping (joint 1)
for m in [0.1,0.05,0.02,-0.1,0.5,-1.0]:
    env=make_env(); obs,_=env.reset(seed=0)
    a=np.zeros(11,dtype=np.float32); a[3]=m
    q=[obs[19]]
    for i in range(15): obs,r,te,tu,_=env.step(a); q.append(obs[19])
    ss=np.diff(q)[-5:].mean()
    print(f"arm mag {m}: steady rate {ss:.5f} ratio {ss/m:.4f}")
    env.close()

# joint limits: drive each joint 300 steps at +0.1 then -0.1
env=make_env()
for j,d in enumerate(range(3,10)):
    for sgn in [1,-1]:
        obs,_=env.reset(seed=0)
        a=np.zeros(11,dtype=np.float32); a[d]=0.1*sgn
        last=obs[19+j]
        for i in range(220):
            obs,r,te,tu,_=env.step(a)
            if abs(obs[19+j]-last)<1e-5 and i>5: break
            last=obs[19+j]
        print(f"joint{j+1} sgn{sgn:+d}: stops at {obs[19+j]:.4f} after {i+1} steps")
env.close()
