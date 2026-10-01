import numpy as np, sys
from env_client import make_env
from approach import GeneratedApproach, Robot
s=int(sys.argv[1])
env=make_env(); obs,info=env.reset(seed=s)
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
n=0
while n<120:
    a=ap.get_action(obs)
    if ap.phase=="open":
        R=Robot(obs)
        if R.holding:
            b=R.blocks[R.held_name()]
            print(n,"open attempt, held",R.held_name(),np.round(b[:7],4),"cell",np.round(ap.cell,4))
            for k,bb in R.blocks.items():
                if bb[7]<0.5: print("   other",k,np.round(bb[:3],4),"yaw",round(float(np.arctan2(2*bb[6]*bb[5],1-2*bb[5]**2)),3))
    obs,r,term,trunc,info=env.step(a); n+=1
    if term: print("TERM",n);break
env.close()
