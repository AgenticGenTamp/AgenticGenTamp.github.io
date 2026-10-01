import sys, numpy as np
from env_client import make_env
from probe_term_lib import *
seed=int(sys.argv[1]); cnt=int(sys.argv[2]); y0=float(sys.argv[3]); y1=float(sys.argv[4]); dy=float(sys.argv[5])
x0,x1=float(sys.argv[6]),float(sys.argv[7])
env=make_env(); obs,info=env.reset(seed=seed,options={'object_count':cnt})
ys=np.arange(y0,y1+1e-9,dy); flip=False; term=False
obs,term,_=move_to(env,obs,np.array([x0,ys[0]]),maxsteps=200)
if term: print('TERM en route',np.round(rxy(obs),3),flush=True)
for y in ys:
    if term: break
    for xt in ([x1,x0] if flip else [x0,x1]):
        obs,term,_=move_to(env,obs,np.array([xt,y]),maxsteps=200)
        if term: break
    if term: break
    flip=not flip
    obs,term,_=move_to(env,obs,np.array([rxy(obs)[0],y+dy]),maxsteps=20)
print(f'seed{seed} cnt{cnt} band[{y0},{y1}] term={term} rend={np.round(rxy(obs),3)}',flush=True)
env.close()
