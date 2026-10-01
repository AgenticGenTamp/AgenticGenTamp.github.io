import sys, numpy as np
from env_client import make_env
from approach import GeneratedApproach
from ctl import rget, oget
seed=int(sys.argv[1])
env = make_env()
ap = GeneratedApproach(env.action_space, env.observation_space, {})
obs, info = env.reset(seed=seed); ap.reset(obs,info)
last=None
for i in range(env.max_steps):
    a = ap.get_action(obs)
    ph=ap.phase
    obs, r, term, trunc, info = env.step(a)
    if ph!=last or i%100==0:
        print(i, ph, "robot",round(rget(obs,'x'),2),round(rget(obs,'y'),2),round(rget(obs,'theta'),2),round(rget(obs,'arm_joint'),2),
              "hook",round(oget(obs,'hook','x'),2),round(oget(obs,'hook','y'),2),round(oget(obs,'hook','theta'),2),int(oget(obs,'hook','held')),
              "blk",round(oget(obs,'target_block','x'),2),round(oget(obs,'target_block','y'),2))
        last=ph
    if term: print("SOLVED",i); break
env.close()
