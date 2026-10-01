import sys; sys.path.insert(0,'.')
import numpy as np
from env_client import make_env
from approach import *
seed=int(sys.argv[1]); upto=int(sys.argv[2])
env=make_env(); obs,info=env.reset(seed=seed)
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
for i in range(upto):
    a=ap.get_action(obs); obs,*_=env.step(a)
cfg=ap._parse(obs)[0]
a=ap.get_action(obs)
prev=cfg
print("cur",np.round(cfg,2))
for w,g in ap.plan:
    print(nsteps(prev,w), np.round(w,2), g); prev=w
from gridroute import grid_path
goal=ap.plan[-1][0]
gp=grid_path(cfg[:3],goal[:3],base_ok)
print("grid", [tuple(np.round(b,2)) for b in gp] if gp else None)
cfg_,blocks,holding,held,gtf,gq=ap._parse(obs)
world=World(blocks)
k=len(gp); dq=cfg_diff(cfg,goal)
seq=[cfg]
for i,b in enumerate(gp[:-1],start=1):
    c=cfg+dq*(i/k); c[0],c[1],c[2]=b; seq.append(c)
seq.append(goal)
for i,c in enumerate(seq[1:]):
    print(i, ap._config_ok(c,world,None,None,False), ap._config_ok(c,world,None,None,False,margin=0.0))
w0=World({})
for i,c in enumerate(seq[1:]):
    print(i, ap._config_ok(c,w0,None,None,False,margin=0.0), np.round(fk_full(c[:3],c[3:])[3],2))
