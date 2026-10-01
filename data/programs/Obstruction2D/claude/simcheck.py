import numpy as np, sys
from env_client import make_env
import approach as A
env=make_env(); ap=A.GeneratedApproach(env.action_space, env.observation_space, {})
for seed in [42,3,5,7,0,1,2,4]:
    obs,info=env.reset(seed=seed)
    ap.reset(obs,info)
    a=ap.get_action(obs); pred=getattr(ap,'expected_steps',None)
    n=0
    for t in range(1000):
        obs,r,term,tr,info=env.step(np.asarray(a,dtype=np.float32)); n+=1
        if term: break
        a=ap.get_action(obs)
    print(seed,"pred",pred,"actual",n)
env.close()
