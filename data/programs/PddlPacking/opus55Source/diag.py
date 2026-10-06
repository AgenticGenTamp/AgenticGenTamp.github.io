import sys, time, numpy as np
from robocode.environments.pr2_tamp_variable_count_env import PR2PackedVariableCountEnv as E
import approach as A
k=int(sys.argv[1]); seed=int(sys.argv[2])
env=E([k],[k]); obs,info=env.reset(seed=seed,options={'object_count':k})
ap=A.GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
flat,_=A.obs_to_flat(obs)
for b in A.block_info(flat,k): print('block', np.round(b['pos'],3), round(A.quat_yaw(b['quat']),2))
for t in range(200):
    a=ap.get_action(obs)
    obs,r,term,_,_=env.step(a)
    if term: print('done',t+1); break
    if not np.any(a[:10]) and a[10]==0:
        flat,_=A.obs_to_flat(obs)
        print('STUCK at',t,'robot',np.round(flat[:12],2))
        for b in A.block_info(flat,k): print('block', np.round(b['pos'],3), round(A.quat_yaw(b['quat']),2), b['held'], ap._block_done(b,None))
        break
