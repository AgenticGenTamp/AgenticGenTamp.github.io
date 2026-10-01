import sys,os,numpy as np
from env_client import make_env
from approach import GeneratedApproach
env=make_env()
for seed in [0,1,2]:
    obs,info=env.reset(seed=seed)
    ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
    for t in range(200):
        a=ap.get_action(obs); d=ap._parse(obs)
        if ap.phase in('grasp_in','grasp_close'): print(seed,t,ap.phase,round(float(d['gap']),3),d['held'],'hw',round(ap.hw,3))
        if ap.phase=='man': break
        obs,r,term,tr,_=env.step(a)
