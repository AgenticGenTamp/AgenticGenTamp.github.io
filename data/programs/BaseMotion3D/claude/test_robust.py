import numpy as np
import approach as A
from env_client import make_env
A.GeneratedApproach.WALL_Y=-1.2   # deliberately wrong model
A.GeneratedApproach.WALL_X_LO=-2.5
A.GeneratedApproach.WALL_X_HI=2.5
env=make_env()
bad=[];steps=[]
for s in range(400):
    o,i=env.reset(seed=s); t0=o[19:21].copy()
    if t0[1]>-1.25: continue
    ap=A.GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(o,i)
    t=False
    for k in range(env.max_steps):
        o,r,t,tr,_=env.step(ap.get_action(o))
        if t or tr: break
    steps.append(k+1)
    if not t: bad.append((s,t0.round(3)))
print("n",len(steps),"mean",np.mean(steps),"max",np.max(steps),"fails",bad[:10], len(bad))
