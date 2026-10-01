import sys, numpy as np
from env_client import make_env
from prototype_agent import GeneratedApproach

seeds=range(20) if len(sys.argv)<2 else map(int,sys.argv[1:])
for seed in seeds:
 e=make_env();s,info=e.reset(seed=seed);p=GeneratedApproach(e.action_space,e.observation_space,{ });p.reset(s,info)
 min_d=np.linalg.norm(s[20:22]-s[29:31]); grasp=False
 for t in range(e.max_steps):
  s,r,term,trunc,info=e.step(p.get_action(s));grasp |= s[6]>.5
  min_d=min(min_d,np.linalg.norm(s[20:22]-s[29:31]))
  if term or trunc:break
 print(seed,'OK' if term else 'FAIL','steps',t+1,'phase',p.phase,'grasp',grasp,'dist',round(float(np.linalg.norm(s[20:22]-s[29:31])),3),'min',round(float(min_d),3))
 e.close()
