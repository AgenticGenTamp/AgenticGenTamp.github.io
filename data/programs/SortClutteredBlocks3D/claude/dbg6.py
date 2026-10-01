import numpy as np
from env_client import make_env
import approach as A
env=make_env(); obs,info=env.reset(seed=0, options={'object_count':4})
ap=A.GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
for t in range(150):
    a=ap.get_action(obs)
    b,q,g=A.robot_state(obs)
    if t>=140:
        gw,p=ap.gripper_world(b,q)
        print(t, ap.phase,'b',np.round(b,4),'gw',np.round(gw,4),'a012',np.round(a[:3],4),'pf',np.round(ap.p_f,4))
    obs,r,te,tr,i=env.step(a)
b,q,g=A.robot_state(obs)
print('now force base move -y')
for t in range(5):
    a=np.zeros(11,dtype=np.float32); a[1]=-0.05
    obs,r,te,tr,i=env.step(a); b,q,g=A.robot_state(obs); print(np.round(b,4))
env.close()
