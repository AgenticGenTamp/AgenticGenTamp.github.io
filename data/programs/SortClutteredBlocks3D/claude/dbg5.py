import numpy as np
from env_client import make_env
import approach as A
from fk import fk
env=make_env(); obs,info=env.reset(seed=0, options={'object_count':4})
ap=A.GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
for t in range(100):
    a=ap.get_action(obs)
    b,q,g=A.robot_state(obs)
    if t>=80:
        gw,p=ap.gripper_world(b,q)
        bxy = np.array(ap.target[:2] if ap.target is not None else A.obj_pos(obs,'cube1')[:2]) - A.rot2(b[2])@p[:2]
        print(t, 'b',np.round(b,4),'btgt',np.round(bxy,4),'a012',np.round(a[:3],4))
    obs,r,te,tr,i=env.step(a)
env.close()
