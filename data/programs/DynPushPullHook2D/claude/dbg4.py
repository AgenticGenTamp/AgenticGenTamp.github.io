import sys, numpy as np
from env_client import make_env
from approach import GeneratedApproach
from ctl import rget, oget, act
seed=int(sys.argv[1]); N=int(sys.argv[2])
env = make_env(); ap = GeneratedApproach(env.action_space, env.observation_space, {})
obs, info = env.reset(seed=seed); ap.reset(obs,info)
for i in range(N):
    obs,r,term,tr,info = env.step(ap.get_action(obs))
    if term: print("SOLVED",i); break
def show(lbl,a,k=3):
    global obs
    x0,y0=rget(obs,'x'),rget(obs,'y'); b0=oget(obs,'target_block','y')
    for _ in range(k): obs,r,term,tr,info=env.step(a)
    print(lbl,"dx",round(rget(obs,'x')-x0,4),"dy",round(rget(obs,'y')-y0,4),"blk dy",round(oget(obs,'target_block','y')-b0,4),"term",term)
show("dy only", act(dy=-0.05))
show("dx-0.02,dy-0.05", act(dx=-0.02,dy=-0.05))
show("dy only again", act(dy=-0.05))
env.close()
