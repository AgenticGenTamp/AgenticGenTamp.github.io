import numpy as np
from env_client import make_env
np.set_printoptions(precision=4, suppress=True)
env = make_env(); o,_=env.reset(seed=0)
tx,ty,tz = o[19],o[20],o[21]
print("target",tx,ty,tz)
# move base toward target
for i in range(200):
    a=np.zeros(11)
    a[0]=np.clip(tx-o[0],-0.4,0.4)
    a[1]=np.clip(ty-o[1],-0.4,0.4)
    o,r,t,tr,inf=env.step(a)
    if t or tr:
        print("done step",i,t,tr,o[:3]); break
else:
    print("no term. final base",o[:3], "dist", np.hypot(o[0]-tx,o[1]-ty))
