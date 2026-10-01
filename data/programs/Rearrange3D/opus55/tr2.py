import sys, numpy as np, approach
from env_client import make_env
from approach import GeneratedApproach
np.set_printoptions(precision=3,suppress=True,linewidth=200)
env=make_env(); obs,info=env.reset(seed=int(sys.argv[1]))
ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(obs,info)
t0,t1=int(sys.argv[2]),int(sys.argv[3])
for t in range(t1):
    a=ap.get_action(obs)
    if t>=t0:
        sg=ap.seg; tr=sg.get('traj') if sg else None
        k=sg['k']-1 if sg else 0
        err=np.max(np.abs(tr[min(k,len(tr)-1)]-obs[96:103])) if tr else 0
        print(t,'k',sg['k'] if sg else None,'n',len(tr) if tr else 0,'qerr %.3f'%err,'base',obs[93:96],'cmd',ap.base_cmd,'a',a[:3])
    obs,r,te,tr_,info=env.step(a)
    if te: print('term',t);break
