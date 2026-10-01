from env_client import make_env
import numpy as np

env=make_env()
for y in [-1.88,-1.90,-1.91,-1.92,-1.94,-1.96,-1.98,-2.,-2.02,-2.04,-2.06,-2.08,-2.10,-2.12,-2.2,-2.3,-2.4,-2.5]:
    s,_=env.reset(seed=779)
    for xy in [[-2.,0],[-2.,y],[-.42034349,y]]:
        for i in range(15):
            a=np.zeros(11,dtype=np.float32)
            a[:2]=np.clip(np.array(xy)-s[:2],-.4,.4)
            prev=s.copy()
            s,r,done,trunc,_=env.step(a)
            if done or np.linalg.norm(s[:2]-xy)<1e-5 or np.linalg.norm(s[:2]-prev[:2])<1e-6:break
        if done:break
    print(y,s[:2].tolist(),done,flush=True)
env.close()
