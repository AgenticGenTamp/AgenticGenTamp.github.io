import numpy as np,sys
from env_client import make_env
from approach import GeneratedApproach
env=make_env(); ap=GeneratedApproach(env.action_space,env.observation_space,{})
o,i=env.reset(seed=int(sys.argv[1])); ap.reset(o,i)
for t in range(1000):
    a=ap.get_action(o); o,r,te,tr,_=env.step(a)
    if t<120: print(t,np.round(a[:3],3),np.round(o[:3],3))
    if te: break
print(t,te)
