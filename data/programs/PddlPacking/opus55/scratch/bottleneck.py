import sys; sys.path.insert(0,'.')
import numpy as np
from env_client import make_env
from approach import GeneratedApproach
cnt={'base_xy':0,'base_yaw':0,'arm':0,'idle':0}; joint=np.zeros(7)
for seed in range(int(sys.argv[1]),int(sys.argv[2])):
    env=make_env(); obs,info=env.reset(seed=seed)
    ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
    for i in range(200):
        a=ap.get_action(obs); obs,r,term,trunc,_=env.step(a)
        m=np.abs(a[:10]); 
        if m.max()<1e-6: cnt['idle']+=1
        elif m[:2].max()>=m.max()-1e-6: cnt['base_xy']+=1
        elif m[2]>=m.max()-1e-6: cnt['base_yaw']+=1
        else: cnt['arm']+=1; joint[np.argmax(m[3:])]+=1
        if term or trunc: break
print(cnt, joint)
