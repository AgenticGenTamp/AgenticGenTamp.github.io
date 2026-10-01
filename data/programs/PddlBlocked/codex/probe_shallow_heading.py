"""Full-episode tests for a more collision-tolerant shallow-SE heading."""
from env_client import make_env
from approach import GeneratedApproach
import math
import numpy as np

for seed in [79,101,363]:
 for theta in [.20,.21]:
  env=make_env();s,i=env.reset(seed=seed);p=GeneratedApproach(env.action_space,env.observation_space,{});p.reset(s,i)
  p.theta=theta;c,z=math.cos(theta),math.sin(theta)
  p.off=np.array([c*p.OFF[0]-z*p.OFF[1],z*p.OFF[0]+c*p.OFF[1]]);p.west_branch=True
  t=p.target(p.blocker);p.approach_route=[np.array([3.4,-.98]),np.array([t[0],-.98]),t] if t[0]>3.65 else [t]
  for step in range(env.max_steps):
   s,_,done,truncated,_=env.step(p.get_action(s))
   if done or truncated:break
  print(seed,theta,'done',done,'steps',step+1,'stage',p.stage)
  env.close()
