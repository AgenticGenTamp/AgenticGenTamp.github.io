import sys, math, numpy as np
from env_client import make_env
from approach import GeneratedApproach
seed=52; T=323
env=make_env(); obs,info=env.reset(seed=seed)
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
for t in range(T):
    a=ap.get_action(obs); obs,*_=env.step(a)
def show(o):
    ap._parse(o); b=ap.blocks['block3']; print(round(ap.arm,3), round(ap.ry,3), np.round(b['poly'][0],3), ap.vac)
show(obs)
for act in [[0,0,0,-0.1,1],[0,0,0,0,1],[0,0,0,-0.05,1],[0,0,0,-0.03,1],[0,0,0,-0.02,1]]:
    obs,*_=env.step(np.array(act,dtype=np.float32)); show(obs)
