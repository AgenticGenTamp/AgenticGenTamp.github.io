import numpy as np
from env_client import make_env
np.set_printoptions(precision=5, suppress=True)
INIT=np.array([0,-0.34907,3.14159,-2.54818,0,-0.87266,1.5708])

# magnitude scaling on base dims + clipping
for d,mags in [(0,[0.1,0.05,0.02,-0.1,0.5,1.0]),(2,[0.1,0.05,-0.1,0.5])]:
    for m in mags:
        env=make_env(); obs,_=env.reset(seed=0)
        a=np.zeros(11,dtype=np.float32); a[d]=m
        b0=obs[16:19].copy()
        for i in range(5): obs,r,te,tu,_=env.step(a)
        per=(obs[16:19]-b0)/5
        print(f"dim{d} mag{m}: per-step base delta {per.round(5)} ratio {round(float(per[d]/m),4) if m else 0}")
        env.close()

# does arm move when base moves?
env=make_env(); obs,_=env.reset(seed=0)
a=np.zeros(11,dtype=np.float32); a[0]=0.1
for i in range(5): obs,r,te,tu,_=env.step(a)
print("arm after base move, diff from init:", (obs[19:26]-INIT).round(6))
env.close()
