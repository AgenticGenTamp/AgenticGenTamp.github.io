import numpy as np
from env_client import make_env
from approach import GeneratedApproach
env=make_env()
for s in range(60):
    o,i=env.reset(seed=s); t0=o[19:22].copy()
    ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(o,i)
    for k in range(60):
        o,r,t,tr,_=env.step(ap.get_action(o))
        if t or tr: break
    n=int(np.ceil(max(abs(t0[0]),abs(t0[1]))/0.4))
    if k+1>n: print(s, t0.round(3), "steps",k+1,"opt",n,"term",t, "finalbase",o[:2].round(4))
