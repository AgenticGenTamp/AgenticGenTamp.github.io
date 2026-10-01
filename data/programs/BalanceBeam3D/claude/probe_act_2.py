import numpy as np
from env_client import make_env
np.set_printoptions(precision=5, suppress=True)

for d in [0,1,2]:
    env = make_env(); obs,_ = env.reset(seed=0)
    a = np.zeros(11, dtype=np.float32); a[d] = 0.1
    tr = [obs[16:19].copy()]
    for i in range(20):
        obs,r,te,tu,_ = env.step(a); tr.append(obs[16:19].copy())
    tr = np.array(tr)
    d16 = np.diff(tr, axis=0)
    print("dim",d,"start",tr[0],"end",tr[-1])
    print("  per-step delta first5", d16[:5].round(5).tolist())
    print("  per-step delta last3", d16[-3:].round(5).tolist())
    print("  arm unchanged?", np.abs(obs[19:26]-np.array([0,-0.34907,3.14159,-2.54818,0,-0.87266,1.5708])).max()<1e-3, "r",r)
    env.close()
