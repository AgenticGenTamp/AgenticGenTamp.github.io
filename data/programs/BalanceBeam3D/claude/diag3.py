import numpy as np, sys
from env_client import make_env
import approach as A
from approach import GeneratedApproach
seed=int(sys.argv[1])
env=make_env(); obs,info=env.reset(seed=seed)
ap=GeneratedApproach(env.action_space, env.observation_space, {})
ap.reset(obs,info)
np.set_printoptions(precision=3,suppress=True)
print("seed",seed,"ss",obs[38:41],"blk",obs[0:2],obs[54:56],obs[70:72])
for t in range(1000):
    a=ap.get_action(obs); obs,r,te,tr,_=env.step(a)
    if t%100==0 or t==999:
        R=A.quat_mat(obs[41:45]); tilt=np.degrees(np.arccos(min(1,abs(R[2,2]))))
        ls=[np.round(A.beam_local(obs,obs[b:b+3]),3) for b in A.BLOCKS]
        print(t,"tilt%.2f"%tilt,"grip",obs[26],"local",ls,flush=True)
    if te: print("TERM",t); break
env.close()
