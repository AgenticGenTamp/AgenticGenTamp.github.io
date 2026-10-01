"""Dense online search around the table-facing initial arm posture."""

import numpy as np
from env_client import make_env


env=make_env(); s,_=env.reset(seed=0); rng=np.random.default_rng(9)
robot=s.get_object_from_name('robot')
lo=np.array([-1.5,-1.6,-3.8,-3.0,-1.3,-1.8,.2])
hi=np.array([ 1.5, .8,-2.5,-1.0, 1.3, .4,2.8])
for trial in range(490):
    target=rng.uniform(lo,hi)
    robot=s.get_object_from_name('robot')
    cur=np.array([s.get(robot,f'joint_{i}') for i in range(1,8)])
    move=np.zeros(11,dtype=np.float32); move[3:10]=np.clip(target-cur,-.4,.4); move[10]=1
    s,_,term,trunc,_=env.step(move)
    close=np.zeros(11,dtype=np.float32); close[10]=-1
    s,_,term,trunc,_=env.step(close); robot=s.get_object_from_name('robot')
    if s.get(robot,'grasp_active')>.5:
        q=[s.get(robot,f'joint_{i}') for i in range(1,8)]
        print('GRASP',trial,q); break
    if term or trunc: break
else: print('NO_GRASP')
print('steps',2*(trial+1)); env.close()
