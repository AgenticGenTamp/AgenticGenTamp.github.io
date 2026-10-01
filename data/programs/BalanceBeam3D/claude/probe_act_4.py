import numpy as np
from env_client import make_env
np.set_printoptions(precision=5, suppress=True)
INIT=np.array([0,-0.34907,3.14159,-2.54818,0,-0.87266,1.5708])
for d in range(3,10):
    j=d-3
    env=make_env(); obs,_=env.reset(seed=0)
    a=np.zeros(11,dtype=np.float32); a[d]=0.1
    tr=[obs[19:26].copy()]
    for i in range(20):
        obs,r,te,tu,_=env.step(a); tr.append(obs[19:26].copy())
    tr=np.array(tr); dj=np.diff(tr[:,j])
    other=np.abs(tr[-1]-tr[0]); other[j]=0
    print(f"a[{d}]->joint{j+1}: start {tr[0,j]:.4f} end {tr[-1,j]:.4f} tot {tr[-1,j]-tr[0,j]:+.4f}")
    print(f"   steps1-4 {dj[:4].round(5)} last3 {dj[-3:].round(5)} max_other_joint_drift {other.max():.5f} base {obs[16:19].round(4)}")
    env.close()
