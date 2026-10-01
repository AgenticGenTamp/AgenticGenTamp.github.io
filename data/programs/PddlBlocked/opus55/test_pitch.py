import sys, numpy as np
from env_client import make_env
from envutil import Sim
from kin import ik, fk_world
from stages import cart_path
import kin
seed=int(sys.argv[1]); pitch=float(sys.argv[2])
env=make_env(); S=Sim(env,env.reset(seed=seed)[0])
blk=S.block('blocker'); g0=S.block('green0')
d=np.r_[g0[:2]-blk[:2],0]; d/=np.linalg.norm(d)
d3=np.r_[d[:2]*np.cos(pitch), -np.sin(pitch)]
up=np.array([0,0,1.0]); zt=up-(up@d3)*d3; zt/=np.linalg.norm(zt)
cb=np.array([blk[0],blk[1],0.8+float(sys.argv[3])])
b0=np.array([3.75,blk[1]-0.15,0.0])
b,q,err=ik(cb-0.12*d3+0.05*up,d3,b0,S.q(),free_base=True,base_nom=b0,zaxis=zt)
print('ik err',err, b.round(2))
print('move', S.moveto(b,S.q()), S.moveto(b,q))
for t in np.linspace(0,1,7)[1:]:
    p=cb-0.12*d3+0.05*up*(1-t)+0.12*d3*t
    _,q,err=ik(p,d3,b,q,free_base=False,zaxis=zt,n_restarts=0)
    r=S.moveto(b,q)
    if not r: print('blocked at t',t, fk_world(b,S.q())[0].round(3)); break
S.step(np.r_[np.zeros(10),-1]); print('held', S.block('blocker')[7])
_,q,err=ik(cb+0.1*up,d3,b,q,free_base=False,zaxis=zt,n_restarts=0); print('lift',S.moveto(b,q),S.block('blocker')[:3].round(3))
