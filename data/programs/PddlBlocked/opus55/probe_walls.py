from env_client import make_env
import numpy as np, sys
from kin import *
from envutil import Sim
from planner import make_plan
seed=int(sys.argv[1]) if len(sys.argv)>1 else 0
env = make_env()
S=Sim(env,env.reset(seed=seed)[0])
blk=S.block('blocker'); g0=S.block('green0')
wps=make_plan(S.base(),S.q(),blk,g0,g0[2])
for w in wps[:3]:
    S.moveto(w['base'],w['q'],maxd=w['maxd'])
    if w['grip']!=0: S.step(np.r_[np.zeros(10),w['grip']])
print('grasp',S.rget('grasp_active'))
d=np.r_[g0[:2]-blk[:2],0]; d/=np.linalg.norm(d); perp=np.array([-d[1],d[0],0])
cb=np.r_[blk[:2],g0[2]+0.002]; b=S.base(); q0=S.q()
def go(p):
    global q
    _,qq,err=ik(p,d,b,S.q(),free_base=False,elbow_min=0.82)
    ok=S.moveto(b,qq,maxd=0.02)
    if not ok: print("  fail dq",(qq-S.q()).round(3),"err",round(err,5))
    return ok
for back in [0.0,0.04,0.08]:
  for sgn in [1,-1]:
    go(cb); go(cb-back*d)
    last=0
    for k in range(1,25):
        if not go(cb-back*d+sgn*0.01*k*perp): break
        last=k
    print('back',back,'side',sgn,'free lateral',last*0.01)
    go(cb-back*d); go(cb)
go(cb)
last=0
for k in range(1,20):
    if not go(cb+0.005*k*d): break
    last=k
print('free forward toward g0', last*0.005, 'gap between faces', 0.15-0.07)
