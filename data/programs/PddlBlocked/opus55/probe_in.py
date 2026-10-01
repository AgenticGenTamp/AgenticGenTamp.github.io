from env_client import make_env
import numpy as np, sys, time
from kin import *
from envutil import Sim
from planner import make_plan
seed=int(sys.argv[1]) if len(sys.argv)>1 else 0
dz=float(sys.argv[2]) if len(sys.argv)>2 else 0.0
env = make_env()
S=Sim(env,env.reset(seed=seed)[0])
blk=S.block('blocker'); g0=S.block('green0')
wps=make_plan(S.base(),S.q(),blk,g0,g0[2])
for w in wps[:6]:
    ok=S.moveto(w['base'],w['q'],maxd=w['maxd'])
    if w['grip']!=0: S.step(np.r_[np.zeros(10),w['grip']])
d=np.r_[g0[:2]-blk[:2],0]; d/=np.linalg.norm(d)
cb=np.r_[blk[:2],g0[2]+dz]; q=S.q(); b=S.base()
for k in range(0,30):
    tgt=cb-0.10*d+0.01*k*d
    _,q,err=ik(tgt,d,b,q,free_base=False,elbow_min=0.85)
    if not S.moveto(b,q,maxd=0.02):
        print('blocked going to dist from blocker centre',round(-0.10+0.01*k,3),'(gap c-c 0.15) err',err); break
p,R=fk_world(S.base(),S.q()); print('tool',p.round(3),'g0',g0[:3].round(3), 'along d dist to g0', round(float((g0[:3]-p)@d),3))
