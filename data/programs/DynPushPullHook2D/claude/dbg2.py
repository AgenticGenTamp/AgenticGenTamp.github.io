import numpy as np
from env_client import make_env
from approach import GeneratedApproach
from ctl import rget, oget, act
env = make_env()
ap = GeneratedApproach(env.action_space, env.observation_space, {})
obs, info = env.reset(seed=7); ap.reset(obs,info)
for i in range(220):
    a = ap.get_action(obs); obs,r,term,tr,info = env.step(a)
    if term: print("solved",i); break
print("stuck state: robot",rget(obs,'x'),rget(obs,'y'),rget(obs,'arm_joint'),"hook",oget(obs,'hook','y'),"blk",oget(obs,'target_block','x'),oget(obs,'target_block','y'),oget(obs,'target_block','width'),oget(obs,'target_block','theta'))
for lbl,a in [("dy-",act(dy=-0.05)),("da-",act(da=-0.1)),("dx-",act(dx=-0.05)),("dx+",act(dx=0.05))]:
    y0=rget(obs,'y'); aj0=rget(obs,'arm_joint'); x0=rget(obs,'x'); b0=oget(obs,'target_block','y')
    for _ in range(5): obs,r,term,tr,info=env.step(a)
    print(lbl,"dy",round(rget(obs,'y')-y0,4),"daj",round(rget(obs,'arm_joint')-aj0,4),"dx",round(rget(obs,'x')-x0,4),"blkdy",round(oget(obs,'target_block','y')-b0,4),"term",term)
env.close()
