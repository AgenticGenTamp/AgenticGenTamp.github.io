"""Test forcing the validated west/southeast branch on stage-0 sweep misses."""
from env_client import make_env
from approach import GeneratedApproach
import numpy as np

for seed in [101,108,117,363]:
 env=make_env();s,i=env.reset(seed=seed);p=GeneratedApproach(env.action_space,env.observation_space,{});p.reset(s,i)
 p.theta=0.;p.off=p.OFF.copy();p.west_branch=True
 t=p.target(p.blocker)
 if t[0]>3.65:
  p.approach_route=[np.array([3.4,-1.6]),np.array([t[0],-1.6]),t]
 else:p.approach_route=[t]
 for step in range(env.max_steps):
  s,_,done,truncated,_=env.step(p.get_action(s))
  if done or truncated:break
 print(seed,'done',done,'steps',step+1,'stage',p.stage,'out',np.round(p.out,3))
 env.close()
