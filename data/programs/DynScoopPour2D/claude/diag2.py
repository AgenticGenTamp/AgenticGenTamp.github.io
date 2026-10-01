from env_client import make_env
import numpy as np, sys
import approach as A
env=make_env(); obs,info=env.reset(seed=int(sys.argv[1]))
ap=A.GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(obs,info)
for t in range(230):
    a=ap.get_action(obs); obs,r,te,tr,info=env.step(a)
    if t%10==0 or t>185:
        print(t,ap.phase,"p=%.3f,%.3f,%.3f,%.2f,%.2f"%tuple(ap.p),"held",ap.held)
