import sys, numpy as np
from env_client import make_env
from approach import GeneratedApproach
sd=int(sys.argv[1]); N=int(sys.argv[2]) if len(sys.argv)>2 else 1000
env = make_env(); obs,info=env.reset(seed=sd)
np.set_printoptions(precision=3, suppress=True)
print('init', obs[0:3], obs[16:18], obs[29:32], obs[12:15])
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
for t in range(N):
    q=len(ap.queue)
    a=ap.get_action(obs)
    if len(ap.queue)>=q and hasattr(ap,'dbg'): print(t, 'pose',obs[0:3],'robot',obs[16:18],'dbg',np.round(ap.dbg,3))
    obs,r,term,trunc,info=env.step(a)
    if term: print('solved',t); break
print('final', obs[0:3])
