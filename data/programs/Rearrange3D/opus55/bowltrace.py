import sys, numpy as np, approach
from env_client import make_env
from approach import GeneratedApproach
np.set_printoptions(precision=3,suppress=True,linewidth=200)
env=make_env(); obs,info=env.reset(seed=int(sys.argv[1]))
ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(obs,info)
every=int(sys.argv[2]) if len(sys.argv)>2 else 5
for t in range(400):
    a=ap.get_action(obs)
    if t%every==0:
        b=obs[0:2]
        print(t,'bowl',obs[0:3],'drink-b',obs[16:18]-b,'%.3f'%obs[18],'can-b',obs[32:34]-b,'%.3f'%obs[34],'tgt',getattr(ap,'cur_tgt',None))
    obs,r,te,tr,info=env.step(a)
    if te: print('term',t);break
