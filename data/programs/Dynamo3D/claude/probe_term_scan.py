import sys, numpy as np
from env_client import make_env
from probe_term_lib import *
seed=int(sys.argv[1]); y0=float(sys.argv[2]); y1=float(sys.argv[3]); dy=float(sys.argv[4])
x0,x1=-1.0,4.0
env=make_env(); obs,info=env.reset(seed=seed)
print('info keys',list(info.keys()) if hasattr(info,'keys') else info,flush=True)
ys=np.arange(y0,y1+1e-9,dy); flip=False; term=False
obs,term,_=move_to(env,obs,np.array([x0,ys[0]]),maxsteps=150)
for y in ys:
    xs=[x1,x0] if flip else [x0,x1]
    for xt in xs:
        obs,term,_=move_to(env,obs,np.array([xt,y]),maxsteps=120)
        if term: break
    if term: break
    flip=not flip
    obs,term,_=move_to(env,obs,np.array([rxy(obs)[0],y+dy]),maxsteps=20)
    if term: break
print(f'seed{seed} band[{y0},{y1}] term={term} rend={np.round(rxy(obs),3)}',flush=True)
env.close()
