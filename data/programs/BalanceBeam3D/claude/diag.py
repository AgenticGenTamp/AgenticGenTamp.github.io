import numpy as np, sys
from env_client import make_env
from approach import GeneratedApproach
import approach as A
seed=int(sys.argv[1])
env=make_env(); obs,info=env.reset(seed=seed)
ap=GeneratedApproach(env.action_space, env.observation_space, {})
ap.reset(obs,info)
np.set_printoptions(precision=4,suppress=True)
print("ss",obs[38:41],"blks",obs[0:3],obs[54:57],obs[70:73])
for t in range(1000):
    a=ap.get_action(obs)
    obs,r,te,tr,_=env.step(a)
    if t%50==0:
        print(t,"ee",np.round(A.ee_world(obs),3),"grip",obs[26],"blks",np.round(obs[0:3],3),np.round(obs[54:57],3),np.round(obs[70:73],3),flush=True)
    if te: print("TERM",t); break
print("end blks",obs[0:3],obs[54:57],obs[70:73],"ss",obs[38:45])
env.close()
