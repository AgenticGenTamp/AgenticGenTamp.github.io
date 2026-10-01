import numpy as np
from env_client import make_env
np.set_printoptions(precision=4, suppress=True)
env = make_env()
rot=np.zeros(11,dtype=np.float32); rot[2]=0.1
for nrot,lbl in [(16,'+90deg'),(31,'+180deg')]:
    for idx,v,nm in [(0,-0.1,'a0=-0.1'),(1,0.1,'a1=+0.1'),(1,-0.1,'a1=-0.1')]:
        obs,_=env.reset(seed=0)
        for _ in range(nrot): obs,_,_,_,_=env.step(rot)
        p0=np.asarray(obs)[93:96].copy()
        act=np.zeros(11,dtype=np.float32); act[idx]=v
        for t in range(10): obs,_,_,_,_=env.step(act)
        p1=np.asarray(obs)[93:96]
        print(f"{lbl} theta={p0[2]:.3f} {nm}: delta={p1-p0}")
env.close()
