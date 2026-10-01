import sys, json, numpy as np, approach
from env_client import make_env
from approach import GeneratedApproach
env=make_env(); obs,info=env.reset(seed=int(sys.argv[1]))
ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(obs,info)
T=int(sys.argv[2])
for t in range(T):
    obs,r,te,tr,info=env.step(ap.get_action(obs))
json.dump([float(x) for x in obs],open(sys.argv[3],'w'))
