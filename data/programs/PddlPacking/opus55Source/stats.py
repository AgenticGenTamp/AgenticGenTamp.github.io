import sys, numpy as np
from robocode.environments.pr2_tamp_variable_count_env import PR2PackedVariableCountEnv as E
import approach as A
names=['bx','by','bth','pan','lift','uroll','elbow','froll','wflex','wroll']
k=int(sys.argv[1])
for seed in range(int(sys.argv[2])):
    env=E([k],[k]); obs,info=env.reset(seed=seed,options={'object_count':k})
    ap=A.GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
    line=[]
    for t in range(300):
        had=len(ap.plan)
        f0,_=A.obs_to_flat(obs)
        a=ap.get_action(obs)
        if had==0:
            n=len(ap.plan)+1
            tot=np.sum([x[:10] for x in [a]+ap.plan],axis=0)
            j=np.argmax(np.abs(tot)); line.append(f'{n}:{names[j]}{abs(tot[j]):.2f}')
        obs,r,term,_,_=env.step(a)
        if term: break
    print(seed, t+1, ' '.join(line))
