import sys, numpy as np
from env_client import make_env
from envutil import Sim
import planner
from planner import make_plan, jdist, cart_path
from kin import fk_world, wrap
seed=int(sys.argv[1]); sides=[float(v) for v in sys.argv[2].split(',')]
env=make_env(); S=Sim(env,env.reset(seed=seed)[0])
blk=S.block('blocker'); g0=S.block('green0'); z=g0[2]
exec(open('run_plan.py').read().split("def follow")[1].join(["def follow",""]) if False else "")
src=open('run_plan.py').read(); fs=src.index("def follow"); fe=src.index("if segs is None")
exec(src[fs:fe])
segs=make_plan(S.base(),S.q(),blk,g0,z)
for w in segs[:3]:
    follow(w['configs'])
    if w['grip']!=0: S.step(np.r_[np.zeros(10),w['grip']])
print('grasp',S.rget('grasp_active'))
planner.FREE_BASE_CHAIN=True
d=np.r_[g0[:2]-blk[:2],0]; d/=np.linalg.norm(d); perp=np.array([-d[1],d[0],0])
cg=np.r_[g0[:2],z]
def tool(): return fk_world(S.base(),S.q())[0]
def go(p, step=0.01):
    qs=cart_path(S.base(),S.q(),tool(),p,d,step=step)
    if qs is None: return None
    return follow(qs)
for l in sides:
    # back off, go lateral at t=-0.25
    go(tool()+[0,0,0.002]-0.0*d)
    p=cg-0.25*d+l*perp+[0,0,0.002]
    ok=go(tool()-0.06*d); ok2=go(p)
    if not ok2: print('l',l,'could not reach start'); continue
    t=-0.25; why="none"
    while t<0.2:
        r=go(cg+(t+0.005)*d+l*perp+[0,0,0.002],step=0.005)
        if not r:
            why = "ikfail" if r is None else "blocked"; break
        t+=0.005
    print("lateral",l,why,'max t (block centre along d from g0)',round(t,3), 'base',S.base().round(2))
    go(cg-0.25*d+l*perp+[0,0,0.002])
