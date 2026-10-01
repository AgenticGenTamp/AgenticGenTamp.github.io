from env_client import make_env
import numpy as np, sys
from kin import *
from envutil import Sim
seed=int(sys.argv[1]) if len(sys.argv)>1 else 0
env = make_env()
S=Sim(env,env.reset(seed=seed)[0])
blk=S.block('blocker'); g0=S.block('green0')
d=g0[:2]-blk[:2]; d=np.r_[d/np.linalg.norm(d),0]
perp=np.array([-d[1],d[0],0])
base=np.array([3.85, g0[1]-0.2, 0.0]); S.moveto(base,S.q())
down=np.array([0,0,-1.])
for zax in [d, perp]:
  q=S.q()
  for z in np.arange(1.05,0.75,-0.01):
    b,q,err=ik([g0[0],g0[1],z],down,base,q,free_base=False,zaxis=zax)
    ok=S.moveto(b,q,maxd=0.05)
    if not ok: print('blocked at',round(z,3),'err',round(err,4)); break
  S.step(np.r_[np.zeros(10),-1.0]); print('grasp',S.rget('grasp_active'), 'tool z',round(z,3))
  if S.rget('grasp_active')>0: break
  S.step(np.r_[np.zeros(10),1.0])
  b,q,err=ik([g0[0],g0[1],1.05],down,base,q,free_base=False,zaxis=zax); S.moveto(b,q,maxd=0.05)
