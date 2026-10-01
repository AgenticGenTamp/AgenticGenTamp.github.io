import numpy as np, sys
from env_client import make_env
from approach import GeneratedApproach, Robot, quat_R, rotz, BLOCK_ORIENTS
import fk
s=int(sys.argv[1])
env=make_env(); obs,info=env.reset(seed=s)
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
n=0
while n<120:
    a=ap.get_action(obs)
    R=Robot(obs)
    if n>=57 and n<110:
        b=R.blocks.get(ap.target)
        pt,Rw,_=R.tool()
        print(n,ap.phase,"tool",np.round(pt,3),"blk",np.round(b[:3],3),"cell",np.round(ap.cell,3),"base",np.round(R.base,2),"act",np.round(a,3))
    obs,r,term,trunc,info=env.step(a); n+=1
    if term: print("TERM",n);break
env.close()
