import numpy as np
from env_client import make_env
from arm import *
mount=MOUNT
env=make_env(); obs,info=env.reset(seed=0)
b,q,g=robot_state(obs)
cb=cube_dict(obs); tgt=cb['cube1']
qt=ik(np.array([0.45,0.0,0.13]), tool_R(0.0), q)
print('q0',np.round(q,3)); print('qt',np.round(qt,3),'fk',np.round(fk(qt)[:3,3],3))
for t in range(200):
    bxy=base_for(tgt[:2], q, mount); bt=np.array([bxy[0],bxy[1],TH])
    a=make_action(b,q,bt,qt,0.0); obs,rw,te,tr,i=env.step(a); b,q,g=robot_state(obs)
    if t%20==0 or t==199:
        print(t,'jerr',np.round(np.max(np.abs((qt-q+np.pi)%(2*np.pi)-np.pi)),3),'btgt',np.round(bt,3),'b',np.round(b,3),'gw',np.round(gripper_world(b,q,mount),3))
env.close()
