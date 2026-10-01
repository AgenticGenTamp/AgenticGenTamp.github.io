import numpy as np
from env_client import make_env
for heading in (0.,.48,np.pi/4,-np.pi/4):
    for x in (-.69,-.71,-.73):
        for y in (-1.97,-1.99):
            env=make_env();s,_=env.reset(seed=779)
            for k in range(4):
                a=np.zeros(11,dtype=np.float32);a[2]=np.clip(heading-s[2],-.4,.4)
                s,*_=env.step(a)
            target=np.array([x,y]);done=False
            for k in range(8):
                a=np.zeros(11,dtype=np.float32);a[:2]=np.clip(target-s[:2],-.4,.4)
                n,*_=env.step(a)
                if np.linalg.norm(n[:2]-target)<.05:done=True;s=n;break
                if np.linalg.norm(n[:2]-s[:2])<1e-7:break
                s=n
            print(round(heading,4),target.tolist(),'reached',done,'pos',s[:3].tolist(),flush=True)
            env.close()
