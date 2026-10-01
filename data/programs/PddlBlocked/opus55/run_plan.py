import time as _tt; _T0=_tt.time()
from env_client import make_env
import numpy as np, sys, time
from kin import *
from envutil import Sim
from planner import make_plan, jdist
seed=int(sys.argv[1]) if len(sys.argv)>1 else 0
env = make_env()
S=Sim(env,env.reset(seed=seed)[0])
blk=S.block('blocker'); g0=S.block('green0')
t=time.time()
segs=make_plan(S.base(),S.q(),blk,g0,g0[2])
print('plan time',round(time.time()-t,2))
def follow(cfgs):
    i=-1
    while True:
        b=S.base(); q=S.q()
        j=i
        while j+1<len(cfgs):
            bb,qq=cfgs[j+1]
            db=np.r_[bb[:2]-b[:2], wrap(bb[2]-b[2])]
            if np.abs(jdist(q,qq)).max()<=0.2+1e-9 and np.abs(db).max()<=0.2+1e-9: j+=1
            else: break
        if j==i and j+1<len(cfgs): j=i+1  # partial step toward next
        bb,qq=cfgs[j]
        dq=jdist(q,qq); db=np.r_[bb[:2]-b[:2], wrap(bb[2]-b[2])]
        delta=np.r_[db,dq]; m=np.abs(delta).max()
        if m<1e-4 and j==len(cfgs)-1: return True
        if m<1e-4: i=j; continue
        a=np.r_[delta/max(1.0,m/0.2),0.0]
        S.step(a)
        if S.done: return True
        if np.abs(np.r_[S.base(),S.q()]-np.r_[b,q]).max()<1e-6:
            print("  BLOCKED at base",b.round(3),"q",q.round(2),"cmd",a[:10].round(3)); return False
        if m<=0.2+1e-9: i=j
if segs is None: print('NO PLAN'); sys.exit()
for w in segs:
    ok=follow(w['configs'])
    if w['grip']!=0: S.step(np.r_[np.zeros(10),w['grip']])
    print(w['name'],'ok',ok,'steps',S.steps,'grasp',S.rget('grasp_active'),'done',S.done)
    if S.done: break
print('WALL',round(_tt.time()-_T0,1),'SEED',seed,'DONE',S.done,'steps',S.steps,'blocker',S.block('blocker')[:3].round(3),'g0',S.block('green0')[:3].round(3))
