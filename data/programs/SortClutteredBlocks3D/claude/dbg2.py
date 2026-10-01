import numpy as np
from env_client import make_env
import approach as A
from fk import fk
env=make_env(); obs,info=env.reset(seed=0, options={'object_count':4})
ap=A.GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
for t in range(200):
    a=ap.get_action(obs); obs,r,te,tr,i=env.step(a)
    if t%10==0:
        b,q,g=A.robot_state(obs); gw,p=ap.gripper_world(b,q)
        print(t, ap.phase, 'gw',np.round(gw,3),'b',np.round(b,3),'p',np.round(p,3),'zb',round(ap.zbias,3))
env.close()
