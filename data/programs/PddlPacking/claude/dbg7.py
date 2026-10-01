import numpy as np, sys
from env_client import make_env
from approach import GeneratedApproach, Robot
import fk
s,n,K=int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3])
env=make_env(); obs,info=env.reset(seed=s,options={"object_count":n})
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
for k in range(K):
    a=ap.get_action(obs); obs,r,t,tr,i=env.step(a)
R=Robot(obs)
print("phase",ap.phase,"q",np.round(R.q,3),"base",np.round(R.base,3))
print("tool",np.round(R.tool()[0],3))
for k,v in R.blocks.items(): print("  ",k,np.round(v[:7],3),"held",v[7])
for j in range(7):
    for sgn in [1,-1]:
        a=np.zeros(11); a[3+j]=0.1*sgn
        o2,_,_,_,_=env.step(np.float32(a)); R2=Robot(o2)
        moved = not np.allclose(R2.q,R.q)
        print("joint",j,"sgn",sgn,"moved",moved)
        if moved:
            a[3+j]=-0.1*sgn; obs,_,_,_,_=env.step(np.float32(a))
for idx in range(3):
    for sgn in [1,-1]:
        a=np.zeros(11); a[idx]=0.1*sgn
        o2,_,_,_,_=env.step(np.float32(a)); R2=Robot(o2)
        moved= not np.allclose(R2.base,R.base)
        print("base",idx,sgn,"moved",moved)
        if moved:
            a[idx]=-0.1*sgn; obs,_,_,_,_=env.step(np.float32(a))
env.close()
