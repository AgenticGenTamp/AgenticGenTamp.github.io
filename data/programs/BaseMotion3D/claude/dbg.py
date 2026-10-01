import numpy as np, approach as A
from env_client import make_env
A.GeneratedApproach.WALL_Y=-1.2; A.GeneratedApproach.WALL_X_LO=-2.5; A.GeneratedApproach.WALL_X_HI=2.5
env=make_env(); o,i=env.reset(seed=80)
ap=A.GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(o,i)
print("target",o[19:21])
for k in range(40):
    a=ap.get_action(o)
    o,r,t,tr,_=env.step(a)
    print(k, np.round(a[:2],4), np.round(o[:2],4), ap.mode, ap.bound, ap.use_true_target, t)
    if t or tr: break
