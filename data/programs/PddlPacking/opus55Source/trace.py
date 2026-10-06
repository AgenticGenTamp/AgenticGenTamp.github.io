import sys, numpy as np
from robocode.environments.pr2_tamp_variable_count_env import PR2PackedVariableCountEnv as E
from approach import GeneratedApproach, obs_to_flat
k=int(sys.argv[1]); s=int(sys.argv[2])
env=E([k],[k]); o,info=env.reset(seed=s, options={'object_count':k})
ap=GeneratedApproach(env.action_space, env.observation_space, None); ap.reset(o,info)
for t in range(200):
    a=ap.get_action(o); o,r,te,tr,_=env.step(a)
    f,_=obs_to_flat(o)
    print(t, 'base', np.round(f[:3],2), 'arm', np.round(f[3:10],2), 'grip', round(float(a[10]),1), 'held', int(f[11]))
    if te or tr: break
