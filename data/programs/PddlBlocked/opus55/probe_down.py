from env_client import make_env
import numpy as np, sys
from kin import *
from envutil import Sim
env = make_env()
S=Sim(env,env.reset(seed=0)[0])
base=np.array([3.85,-0.3,0.0]); S.moveto(base,S.q())
q=S.q(); d=np.array([1.,0,0])
for z in np.arange(1.0,0.2,-0.01):
    b,q,err=ik([4.35,-0.3+0.188,z],d,base,q,free_base=False,elbow_min=0.85)
    ok=S.moveto(b,q,maxd=0.05)
    if round(z*100)%5==0: print(round(z,2),round(err,4),[p.round(3) for p in fk_points(base,S.q())][1:])
    if not ok:
        print('blocked at cmd z',round(z,3),'err',err, 'pts',[p.round(3) for p in fk_points(base,S.q())]); break
print(S.q().round(3))
