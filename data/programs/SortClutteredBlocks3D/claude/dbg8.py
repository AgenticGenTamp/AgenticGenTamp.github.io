import numpy as np
from env_client import make_env
import approach as A
env=make_env(); obs,info=env.reset(seed=0, options={'object_count':4})
ap=A.GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
prev=None
for t in range(200):
    a=ap.get_action(obs)
    b,q,g=A.robot_state(obs); gw,p=ap.gripper_world(b,q)
    if ap.phase in ('down','close','lift') or (prev in ('down','close','lift')):
        print(t, ap.phase,'gw',np.round(gw,4),'cube1',np.round(A.obj_pos(obs,'cube1'),4),'grip',g,'bias',np.round(ap.bias,3))
    prev=ap.phase
    obs,r,te,tr,i=env.step(a)
env.close()
