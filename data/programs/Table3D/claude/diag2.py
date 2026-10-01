import numpy as np, time
from env_client import make_env
import approach as A
seed=89
env=make_env(); obs,info=env.reset(seed=seed)
ap=A.GeneratedApproach(env.action_space,env.observation_space,{})
ap.reset(obs,info)
print("cubes:")
base=ap._base(obs)
for n,p,he in ap._cubes(obs):
    print("  ",n,np.round(p,3),"base-rel",np.round(ap._to_base(p,base),3))
print("order",ap.cube_order,"base_goal",ap.base_goal)
last=None
for i in range(300):
    a=ap.get_action(obs)
    obs,r,t,tr,inf=env.step(a)
    cur=(ap.phase,getattr(ap,'cube_name','-'),ap.attempt_i if ap.attempts else -1,ap.stage)
    if cur!=last:
        print(i,cur,"grasped",ap._grasped(obs),"base",np.round(ap._base(obs),2))
        last=cur
    if t: print("TERM at",i); break
env.close()
