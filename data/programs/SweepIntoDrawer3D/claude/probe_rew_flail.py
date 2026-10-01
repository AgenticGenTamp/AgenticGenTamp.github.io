import numpy as np
from env_client import make_env
np.set_printoptions(precision=3, suppress=True, linewidth=250)
rng=np.random.default_rng(0)
env=make_env(); obs,_=env.reset(seed=0)
c0=obs[:80].reshape(5,16)[:,:3].copy()
rs=[]
for k in range(300):
    a=rng.uniform(-0.1,0.1,11); a[10]=rng.random()
    obs,r,t,tr,i=env.step(a)
    c=obs[:80].reshape(5,16)[:,:3]
    d=np.linalg.norm(c-c0,axis=1)
    rs.append(r)
    if k%20==0 or abs(r+1.0)>1e-6:
        print(k,"r=%.4f"%r,"moved",np.round(d,3),"drawer",np.round(obs[103:109],3),"t",t,tr)
    if t or tr: print("END",t,tr,i); break
print("final cubes",np.round(obs[:80].reshape(5,16)[:,:3],3))
print("rew set",sorted(set(np.round(rs,5)))[:20])
env.close()
