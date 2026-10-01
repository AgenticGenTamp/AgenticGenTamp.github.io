import numpy as np
from env_client import make_env
import approach as A
env=make_env(); obs,info=env.reset(seed=0, options={'object_count':4})
ap=A.GeneratedApproach(None,None,{}); ap.reset(obs,info)
for t in range(220):
    a=ap.get_action(obs)
    b,q,g=A.robot_state(obs); gw,p=ap.gripper_world(b,q)
    if 85<=t<=220 and t%6==0:
        c=A.obj_pos(obs,'cube2')
        print(t, ap.phase, 'gw',np.round(gw,3),'tgt_c2',np.round(c[:2],3),'b',np.round(b,3),'bdes',np.round(ap.base_des(c[:2]),3),'bias',np.round(ap.bias,3),'a01',np.round(a[:2],3))
    obs,r,te,tr,i=env.step(a)
env.close()
