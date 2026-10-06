import sys, numpy as np
from robocode.environments.pr2_tamp_variable_count_env import PR2PackedVariableCountEnv as E
import approach as A
k=int(sys.argv[1]); seed=int(sys.argv[2])
env=E([k],[k]); obs,info=env.reset(seed=seed,options={'object_count':k})
ap=A.GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
for t in range(300):
    had=len(ap.plan)
    f0,_=A.obs_to_flat(obs)
    a=ap.get_action(obs)
    if had==0:
        print('t',t,'macro len',len(ap.plan)+1,'held',f0[11], 'base',np.round(f0[:3],2))
    obs,r,term,_,_=env.step(a)
    f1,_=A.obs_to_flat(obs)
    if a[10]!=0:
        print('   grip',a[10],'-> held',f1[11], [ (i,np.round(b['pos'],3).tolist()) for i,b in enumerate(A.block_info(f1,k)) if b['held'] or np.linalg.norm(b['pos'][:2])<0.2])
    if term: break
print('steps',t+1)
