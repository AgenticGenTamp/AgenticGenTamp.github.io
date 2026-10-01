import numpy as np
from env_client import make_env
from approach import GeneratedApproach, Robot
import fk
env=make_env(); obs,info=env.reset(seed=7)
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
for n in range(63):
    a=ap.get_action(obs); obs,r,t,tr,i=env.step(a)
R=Robot(obs)
print("q",np.round(R.q,3)); print("limits lo",fk.LIMITS[:,0],"hi",fk.LIMITS[:,1])
print("base",np.round(R.base,3),"tool",np.round(R.tool()[0],3))
print("blocks",{k:np.round(v[:3],3) for k,v in R.blocks.items()})
for j in range(7):
    for sgn in [1,-1]:
        a=np.zeros(11); a[3+j]=0.1*sgn
        o2,_,_,_,_=env.step(np.float32(a))
        R2=Robot(o2)
        moved = not np.allclose(R2.q,R.q)
        print("joint",j,"sgn",sgn,"moved",moved)
        if moved:
            a[3+j]=-0.1*sgn
            obs,_,_,_,_=env.step(np.float32(a))
            R=Robot(obs)
            assert np.allclose(Robot(obs).q,R.q)
env.close()
