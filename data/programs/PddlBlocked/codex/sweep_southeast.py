"""Selective regression sweep for no-spare southeast pen orientations."""
from env_client import make_env
from approach import GeneratedApproach
import sys

stop=int(sys.argv[1]) if len(sys.argv)>1 else 500
env=make_env();tested=wins=0
for seed in range(stop):
 s,info=env.reset(seed=seed);names=s.get_object_names()
 p=GeneratedApproach(env.action_space,env.observation_space,{});p.reset(s,info)
 if any(n.startswith('green') and n!='green0' for n in names) or not (p.out[0]>.5 and p.out[1]<-.3):continue
 tested+=1
 for step in range(env.max_steps):
  s,_,done,truncated,_=env.step(p.get_action(s))
  if done or truncated:break
 wins+=int(done)
 if not done:print('FAIL',seed,'out',p.out,'stage',p.stage,'held',p.g(s,'robot','grasp_active'))
print('southeast',wins,'/',tested)
env.close()
