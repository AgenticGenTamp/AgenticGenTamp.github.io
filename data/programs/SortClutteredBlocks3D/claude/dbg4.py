import numpy as np
from env_client import make_env
from arm import robot_state, make_action
env=make_env(); obs,info=env.reset(seed=0)
b,q,g=robot_state(obs); q0=q.copy()
def drive(bt,n=40):
    global b,q,g,obs
    for t in range(n):
        a=make_action(b,q,bt,q0,0.0); obs,r,te,tr,i=env.step(a); b,q,g=robot_state(obs)
    return b.copy()
import math
for ang in [0, 30, 60, 90]:
    # go far out then approach origin along direction
    d=np.array([math.cos(math.radians(ang)), math.sin(math.radians(ang))])
    far = d*1.5
    drive(np.array([far[0],far[1],np.pi]),60)
    got = drive(np.array([0,0,np.pi]),80)
    print('ang',ang,'stalled at',np.round(got,3),'dist',round(float(np.linalg.norm(got[:2])),3))
env.close()
