from env_client import make_env
from ctl import act, rget, oget
import numpy as np
env=make_env()
for seed in [42,0,7]:
    obs,info=env.reset(seed=seed)
    def H(): return (round(oget(obs,'hook','x'),4),round(oget(obs,'hook','y'),4),round(oget(obs,'hook','theta'),4))
    print(seed,"h0",H())
    for _ in range(50): obs,_,_,_,_=env.step(act())
    print(seed,"after 50 noop",H())
env.close()
