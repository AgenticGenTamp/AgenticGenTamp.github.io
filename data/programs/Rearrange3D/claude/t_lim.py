import numpy as np
from env_client import make_env
np.set_printoptions(precision=4, suppress=True)
env = make_env()
res={}
for j in range(7):
    row=[]
    for sgn in [1,-1]:
        obs,_=env.reset(seed=0)
        act=np.zeros(11,dtype=np.float32); act[3+j]=0.1*sgn
        prev=None; val=None
        for t in range(400):
            obs,_,_,_,_=env.step(act); cur=float(np.asarray(obs)[96+j])
            if prev is not None and abs(cur-prev)<1e-5:
                val=cur; break
            prev=cur
        row.append((round(prev if val is None else val,4), t))
    print(f"joint{j+1}: max={row[0]} min={row[1]}")
env.close()
