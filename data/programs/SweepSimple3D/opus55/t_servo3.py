from env_client import make_env
import numpy as np, kin, rutil
from rutil import Bot
env=make_env()
for g in [1.0,1.5,2.0]:
    rutil.ARM_GAIN=g
    obs,_=env.reset(seed=0); b=Bot(env,obs)
    q0=b.q(); qt=q0+np.array([0.3,0.3,0.2,0.3,0.2,0.3,0.2])
    out=[]
    for k in range(40):
        b.act(q_t=qt); out.append(np.max(np.abs(qt-b.q())))
    print(g,np.round(out,4))
