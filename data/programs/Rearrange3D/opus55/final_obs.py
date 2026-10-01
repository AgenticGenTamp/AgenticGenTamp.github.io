import numpy as np, sys, json
from env_client import make_env
from approach import GeneratedApproach
np.set_printoptions(precision=3,suppress=True,linewidth=200)
seed=int(sys.argv[1]); N=int(sys.argv[2]) if len(sys.argv)>2 else 1000
env=make_env(); obs,info=env.reset(seed=seed)
ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(obs,info)
for t in range(N):
    obs,r,te,tr,info=env.step(ap.get_action(obs))
    if te or tr: break
print(t,te,info)
for o in (0,16,32): print(o, obs[o:o+7], obs[o+13:o+16])
json.dump(obs.tolist(),open(f'final_{seed}.json','w'))
