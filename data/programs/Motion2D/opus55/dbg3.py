from env_client import make_env
import numpy as np
env = make_env()
for delta in [0, 2e-5, 5e-5, 1e-4, 3e-4, 1e-3]:
    obs, info = env.reset(seed=159)
    r=obs.get_object_from_name('robot')
    def st(a):
        global obs
        obs,*_=env.step(np.array(a,dtype=np.float32))
    for i in range(5): st([0,0,0.1748,0,0])
    for i in range(20): st([0,0.05,0,0,0])
    y=float(obs.get(r,'y')); goal=2.4-delta
    for k in range(10):
        y=float(obs.get(r,'y')); d=np.clip(goal-y,-0.05,0.05)
        st([0,d,0,0,0])
    y1=float(obs.get(r,'y'))
    for i in range(20): st([0.05,0,0,0,0])
    print(delta, repr(y1), float(obs.get(r,'x')))
