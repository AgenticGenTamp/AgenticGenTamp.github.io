import numpy as np, sys
from common import make_env, robot, obj
import approach
env=make_env([1,2])
n=int(sys.argv[1]); seed=int(sys.argv[2]); T=int(sys.argv[3])
appr=approach.GeneratedApproach(env.action_space, env.observation_space, {})
s,info=env.reset(seed=seed, options={'object_count':n}); appr.reset(s,info)
for i in range(T):
    a=appr.get_action(s)
    s,r,term,_,_=env.step(a)
    if i%10==0 or term:
        base,q=robot(s)
        print(i, 'base',base.round(2),'q',q.round(2),'g',a[10], ' '.join(f'{c}:{obj(s,c)[:3].round(2)}' for c in appr._cubes()))
    if term: break
