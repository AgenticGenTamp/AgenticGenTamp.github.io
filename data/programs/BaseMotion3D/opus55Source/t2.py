import numpy as np, sys, time
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
from approach import GeneratedApproach
env = KinderGeom3DEnv("kinder/BaseMotion3D-v0")
for seed in [int(x) for x in sys.argv[1:]]:
    obs,info=env.reset(seed=seed); t0=time.time()
    ap=GeneratedApproach(env.action_space,env.observation_space,{})
    for t in range(100):
        obs,r,term,trunc,_=env.step(ap.get_action(obs))
        if term or trunc: break
    print(seed, term, t+1, time.time()-t0, obs[:3], obs[19:21], flush=True)
