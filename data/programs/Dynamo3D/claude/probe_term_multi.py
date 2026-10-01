import sys, numpy as np
from env_client import make_env
from probe_term_lib import *
seed=int(sys.argv[1]); tgt=np.array([float(v) for v in sys.argv[2].split(',')])
env=make_env(); obs,info=env.reset(seed=seed)
cs={n:np.round(cxy(obs,n),3).tolist() for n in chairs(obs)}
r0=rxy(obs)
obs,t,n=move_to(env,obs,tgt,maxsteps=200)
print(f'seed{seed} chairs={cs} r0={np.round(r0,3)} tgt={tgt} term={t} steps={n} rend={np.round(rxy(obs),3)}')
env.close()
