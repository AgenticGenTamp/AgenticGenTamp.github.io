from env_client import make_env
import numpy as np, kin
from rutil import Bot
env=make_env(); obs,_=env.reset(seed=0); b=Bot(env,obs)
q0=b.q(); qt=q0+np.array([0,0,0,0,0,0,0.3])
for k in range(30):
    qp=b.q(); b.act(q_t=qt)
    print(k,'err',np.round((qt-b.q())[6],4),'dq',np.round((b.q()-qp)[6],4), 'v',round(b.obs.get(b.R,'vel_arm_joint7'),3), 'base',np.round(b.base(),3))
