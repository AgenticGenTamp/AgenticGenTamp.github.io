from env_client import make_env
import numpy as np

env=make_env()
for theta in [0,.4,.8,1.2,1.5707963,2.,2.4,3.1415926,-.4,-.8,-1.2,-1.5707963]:
    s,_=env.reset(seed=56)
    done=False
    for i in range(30):
        a=np.zeros(11,dtype=np.float32)
        a[:2]=np.clip(s[19:21]-s[:2],-.4,.4)
        a[2]=np.clip(theta-s[2],-.4,.4)
        s,r,done,trunc,_=env.step(a)
        if done:break
    print(theta,done,i+1,s[:3].tolist(),flush=True)
# Move to a point just within the goal radius rather than target center.
for off in [.01,.02,.03,.04,.049]:
    s,_=env.reset(seed=56)
    for i in range(10):
        a=np.zeros(11,dtype=np.float32)
        a[:2]=np.clip(s[19:21]+[0,off]-s[:2],-.4,.4)
        s,r,done,trunc,_=env.step(a)
        if done:break
    print('offset',off,done,s[:3].tolist(),flush=True)
env.close()
