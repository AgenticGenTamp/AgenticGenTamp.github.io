import numpy as np
from env_client import make_env
from arm import robot_state
env=make_env(); obs,info=env.reset(seed=0)
b,q,g=robot_state(obs); q0=q.copy()
for mag in [0.002,0.005,0.008,0.01,0.015,0.02,0.025,0.03,0.04]:
    a=np.zeros(11,dtype=np.float32); a[1]=mag; a[3]=mag
    prev_b=b.copy(); prev_q=q.copy(); dys=[];djs=[]
    for t in range(6):
        obs,r,te,tr,i=env.step(a); b,q,g=robot_state(obs)
        dys.append(round(float(b[1]-prev_b[1]),5)); djs.append(round(float(q[0]-prev_q[0]),5))
        prev_b=b.copy(); prev_q=q.copy()
    print(f"mag {mag}: dbase_y {dys}  dq1 {djs}")
env.close()
