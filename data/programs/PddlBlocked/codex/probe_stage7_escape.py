"""Probe collision-free escape directions after the southeast green lift stalls."""
from env_client import make_env
from approach import GeneratedApproach
import numpy as np

for delta in ([0,-.2],[-.2,0],[-.2,-.2],[0,.2],[-.2,.2],[.2,0]):
  env=make_env();s,i=env.reset(seed=74);p=GeneratedApproach(env.action_space,env.observation_space,{});p.reset(s,i)
  same=0;old=None
  for k in range(120):
    s,_,done,truncated,_=env.step(p.get_action(s))
    pos=p.robot(s)
    if p.stage==7 and old is not None and np.max(np.abs(pos-old))<1e-7:same+=1
    else:same=0
    old=pos.copy()
    if same>=3:break
  before=p.robot(s).copy(); target=before+np.array(delta)
  s,_,_,_,_=env.step(p.motion(s,target,1,lift=True,q4add=.08))
  after=p.robot(s)
  print('delta',delta,'before',before,'after',after,'accepted',np.max(np.abs(after-before))>1e-5,
        'green',[round(p.g(s,'green0','pose_'+x),3) for x in 'xyz'])
  env.close()
