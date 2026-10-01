from env_client import make_env
import numpy as np, sys
from kin import *
from envutil import Sim
env = make_env()
S=Sim(env,env.reset(seed=0)[0])
blk=S.block('blocker'); g0=S.block('green0')
d=g0[:2]-blk[:2]; d=np.r_[d/np.linalg.norm(d),0]
base=np.array([3.85, blk[1]-0.2, 0.0])
S.moveto(base,S.q())
tgt=blk[:3]; pre=tgt-0.15*d
b,q,err=ik(pre+[0,0,0.15],d,base,S.q(),free_base=False)
print('goal q',q.round(3)); print('start q',S.q().round(3))
print('goal pts',[p.round(3) for p in fk_points(base,q)])
ok=S.moveto(b,q,grip=1.0)
print(ok,'blocked q',S.q().round(3))
print('pts',[p.round(3) for p in fk_points(base,S.q())])
