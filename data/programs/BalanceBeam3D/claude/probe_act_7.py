import numpy as np
from env_client import make_env
np.set_printoptions(precision=4, suppress=True)
env=make_env(); obs,_=env.reset(seed=0)
print("grip init", obs[26])
def run(g,n):
    a=np.zeros(11,dtype=np.float32); a[10]=g
    vals=[]
    for i in range(n):
        o,r,te,tu,_=env.step(a); vals.append(float(o[26]))
    return vals
for g in [1.0, 0.0, 0.5, 1.0, 0.25, 0.0]:
    v=run(g,10)
    print(f"grip cmd {g}: traj[:4] {np.round(v[:4],4)} final {v[-1]:.4f}")
# out of range
for g in [2.0, -1.0]:
    v=run(g,10); print(f"grip cmd {g}: final {v[-1]:.4f}")
env.close()
