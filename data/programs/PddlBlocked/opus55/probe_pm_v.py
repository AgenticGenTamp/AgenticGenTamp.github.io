from env_client import make_env
import numpy as np
from kin import ik, fk_world
from envutil import Sim
env=make_env(); S=Sim(env,env.reset(seed=0)[0]); g=S.block('green0')
B=np.array([3.75,0.1,0.0])
for zx in [(1,0,0),(0,1,0)]:
  for p in [(4.3,0.2,1.05),(4.5,0.3,1.05),(4.6,0.4,1.0),(4.5,0.0,0.9)]:
    b,q,e=ik(np.array(p),np.array([0,0,-1.]),B,S.q(),free_base=False,zaxis=zx)
    print(zx,p,round(e,4),q.round(2))
