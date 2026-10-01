import sys, numpy as np
from env_client import make_env
from probe_term_lib import *
seed=int(sys.argv[1]); cnt=int(sys.argv[2])
g=np.array([1.0,0.0]) if cnt==1 else np.array([2.55,2.55])
env=make_env(); obs,info=env.reset(seed=seed,options={'object_count':cnt})
n=0; t=False
for i in range(400):
    p=rxy(obs); d=g-p; nn=np.linalg.norm(d)
    v=d/max(nn,1e-9)*min(0.1,nn)
    obs,rew,t,tr,info=env.step(act(v[0],v[1])); n+=1
    if t or tr or nn<0.01: break
print(f'seed{seed} cnt{cnt} term={t} steps={n} rend={np.round(rxy(obs),3)}',flush=True)
env.close()
