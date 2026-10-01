import numpy as np
from env_client import make_env
import approach as A
env=make_env()
for s in [72,601,1313,560]:
  for rot in [0.785,0.6,0.4]:
    o,_=env.reset(seed=s); tgt=o[19:21].astype(float); g=A.plan_goal(np.zeros(2),tgt)
    # also try goal shifted away from y-edge along x toward outside of the table
    for t in range(8):
        a=np.zeros(11); a[:2]=np.clip(g-o[:2],-0.4,0.4); a[2]=np.clip(rot-o[2],-0.4,0.4)
        o,r,te,*_=env.step(a)
        if te: break
    print(s,rot,te,t+1,np.round(o[:3],3),np.round(tgt,3))
