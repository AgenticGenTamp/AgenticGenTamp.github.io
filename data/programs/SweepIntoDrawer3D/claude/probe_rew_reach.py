import numpy as np
from env_client import make_env
np.set_printoptions(precision=3, suppress=True, linewidth=250)
rng=np.random.default_rng(1)
env=make_env(); obs,_=env.reset(seed=0)
c0=obs[:80].reshape(5,16)[:,:3].copy()
print("base0",np.round(obs[125:128],3),"cubes",np.round(c0,3).tolist())
for k in range(15):
    a=np.zeros(11); a[0]=-0.1
    obs,r,t,tr,_=env.step(a)
print("base after fwd",np.round(obs[125:128],3))
lo=np.array([-2.9,-1.76,-2.9,-3.07,-2.9,-0.02,-2.9]); hi=np.array([2.9,1.76,2.9,-0.07,2.9,3.75,2.9])
recs=[]
for trial in range(40):
    qt=rng.uniform(lo,hi); g=rng.random()
    for k in range(25):
        q=obs[128:135]
        a=np.zeros(11); a[3:10]=np.clip(qt-q,-0.1,0.1); a[10]=g
        obs,r,t,tr,_=env.step(a)
    c=obs[:80].reshape(5,16)[:,:3]
    d=np.linalg.norm(c-c0,axis=1)
    w=obs[147:150]
    if d.max()>0.005 or abs(r+1)>1e-6 or abs(obs[103:109]).max()>0.02:
        print("trial",trial,"r=%.4f"%r,"moved",np.round(d,3),"cubes",np.round(c,3).tolist(),"wiper",np.round(w,3),"drawer",np.round(obs[103:109],3))
    recs.append(r)
print("rewards seen",sorted(set(np.round(recs,4))))
print("final base",np.round(obs[125:128],3),"wiper",np.round(obs[147:150],3))
env.close()
