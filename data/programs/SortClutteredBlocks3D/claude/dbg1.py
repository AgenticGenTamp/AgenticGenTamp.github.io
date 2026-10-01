import numpy as np
from env_client import make_env
import approach as A
from fk import fk, ik
env=make_env(); obs,info=env.reset(seed=0, options={'object_count':4})
ap=A.GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
for t in range(300):
    a=ap.get_action(obs); obs,r,te,tr,i=env.step(a)
    if t%20==0:
        b,q,g=A.robot_state(obs); p=fk(q)[:3,3]
        gw=np.array([b[0]-p[0], b[1]-p[1], A.MOUNT[2]+p[2]])
        print(t, ap.phase, 'gw',np.round(gw,3),'b',np.round(b,3),'p',np.round(p,3))
env.close()
