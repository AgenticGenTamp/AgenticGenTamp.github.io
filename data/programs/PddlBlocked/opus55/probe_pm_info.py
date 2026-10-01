from env_client import make_env
import numpy as np, time
from envutil import Sim
env=make_env()
for s in range(10):
    S=Sim(env,env.reset(seed=s)[0]); g=S.block('green0'); b=S.block('blocker')
    d=g[:2]-b[:2]; d/=np.linalg.norm(d)
    qx,qy,qz,qw=g[3:7]; yaw=np.arctan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz))
    print(s,'g0',g[:3].round(3),'blk',b[:3].round(3),'d',d.round(3),'atan2d',round(np.degrees(np.arctan2(d[1],d[0])),1),'g0yaw',round(np.degrees(yaw),1))
t=time.time(); 
for i in range(20): S.step(np.zeros(11))
print('step time',(time.time()-t)/20)
