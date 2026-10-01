import numpy as np
from env_client import make_env
np.set_printoptions(precision=4, suppress=True)
env=make_env()
for seed in [1,2,7]:
    obs,_=env.reset(seed=seed); p0=np.asarray(obs)[93:96].copy()
    a=np.zeros(11,dtype=np.float32); a[1]=0.1
    for t in range(10): obs,_,_,_,_=env.step(a)
    p1=np.asarray(obs)[93:96]
    a=np.zeros(11,dtype=np.float32); a[0]=-0.1
    for t in range(10): obs,_,_,_,_=env.step(a)
    p2=np.asarray(obs)[93:96]
    print(f"seed{seed} start={p0} dy10={(p1-p0)[1]:.4f} dx10={(p2-p1)[0]:.4f}")
env.close()
