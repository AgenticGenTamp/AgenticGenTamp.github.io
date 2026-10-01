import sys
from env_client import make_env
from approach import GeneratedApproach
from dump import dump
import numpy as np
seed=int(sys.argv[1]); every=int(sys.argv[2]) if len(sys.argv)>2 else 20
env=make_env()
import os; cnt=os.environ.get("COUNT"); obs,info=env.reset(seed=seed, options=({"object_count":int(cnt)} if cnt else None)); dump(obs)
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
for t in range(1000):
    a=ap.get_action(obs)
    if t%every==0 or t<3: print(t,'r',round(ap.rx,3),round(ap.ry,3),round(ap.rth,2),round(ap.arm,3),'st',[round(v,3) for v in ap.stick[:3]],'g',ap.grasped,ap.grasp_stage,'stuck',ap.stuck,'a',np.round(a,3),'btn',[(n,round(x,2),round(y,2)) for n,x,y,_ in ap.buttons])
    obs,r,term,trunc,_=env.step(a)
    if term: print('done',t+1); break
